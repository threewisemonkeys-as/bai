#!/usr/bin/env python3
"""Create an editable native .fig using magnets_learning_compact as its reference.

Setup: bun add --cwd /tmp/grow_figma_compact kiwi-schema@0.5.0
Run: .venv/bin/python offline_learning/scripts/fig_grow_learning_compact.py

Reads the user-supplied native Figma document, its embedded schema and font
outlines, plus the audited Grow evidence. No image generation, Sketch conversion,
Figma MCP, or external Figma document is involved. The input .fig is never edited.
"""
from __future__ import annotations

import argparse
import base64
import collections
import copy
import csv
import datetime
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from offline_learning.validate import run_perceive

REFERENCE = ROOT / 'analysis/learning_example/learning_evolution_fig.fig'
EVIDENCE = ROOT / 'analysis/learning_example/learning_evolution_grow_relations_evidence.json'
OUT = ROOT / 'analysis/learning_example'
IO = ROOT / 'offline_learning/scripts/figma_export/fig_kiwi_io.cjs'
BG, INK, MUTED = '#fcfcfb', '#17212b', '#717b83'
BLUE, GREEN = '#2168b5', '#168057'
BLUE_BG, GREEN_BG, GRAY_BG, RULE = '#eaf2fb', '#e8f4ee', '#f1f3f3', '#dce0e0'
W, H = 1829, 1050
COLS = [23 + i * 286.74 for i in range(4)]
BX, BW = 1097, 717
ROW_Y = [270, 439, 664, 833]
SUBS = '₀₁₂₃'


def gid(g):
    return (g['sessionID'], g['localID'])


def as_bytes(blob):
    return base64.b64decode(blob['$bytes'])


def matrix(n):
    t = n.get('transform', {})
    return Affine2D.from_values(t.get('m00', 1), t.get('m10', 0),
                                t.get('m01', 0), t.get('m11', 1),
                                t.get('m02', 0), t.get('m12', 0))


def transform(x=0, y=0):
    return dict(m00=1, m01=0, m02=x, m10=0, m11=1, m12=y)


def paint(color):
    from matplotlib.colors import to_rgba
    r, g, b, a = to_rgba(color)
    return [dict(type='SOLID', color=dict(r=r, g=g, b=b, a=a), opacity=1,
                 visible=True, blendMode='NORMAL',
                 authoredColor=dict(space='DEFAULT', c0=r, c1=g, c2=b, c3=a))]


def color_of(paints):
    if not paints:
        return None
    p = paints[0]
    c = p['color']
    return (c['r'], c['g'], c['b'], c.get('a', 1) * p.get('opacity', 1))


def rect_commands(x, y, w, h, reverse=False):
    pts = [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
    if reverse:
        pts.reverse()
    return b''.join(bytes([1 if i == 0 else 2]) + struct.pack('<ff', *p)
                    for i,p in enumerate(pts)) + bytes([0])


def decode_path(data):
    """Figma command stream: close=0, move=1, line=2, quadratic=3, cubic=4."""
    offset, vertices, codes, start = 0, [], [], None
    while offset < len(data):
        tag = data[offset]; offset += 1
        if tag == 0:
            if start is not None:
                vertices.append(start); codes.append(MplPath.CLOSEPOLY); start = None
            continue
        count = {1: 2, 2: 2, 3: 4, 4: 6}.get(tag)
        if count is None:
            raise ValueError(f'Unknown geometry command {tag}')
        coords = struct.unpack_from('<'+'f'*count, data, offset); offset += 4*count
        if tag == 1:
            start = coords
        for i in range(0, count, 2):
            vertices.append(coords[i:i+2]); codes.append({1:MplPath.MOVETO,2:MplPath.LINETO,3:MplPath.CURVE3,4:MplPath.CURVE4}[tag])
    return MplPath(vertices, codes) if vertices else None


class Builder:
    def __init__(self, source):
        self.source = source
        self.nodes = []
        self.blobs = copy.deepcopy(source['blobs'])
        self.session = max(n['guid']['sessionID'] for n in source['nodeChanges']) + 1
        self.positions = collections.Counter()
        self.glyphs = {}
        self.fontmeta = {}
        self.text_templates = {}
        self.byid = {gid(n['guid']): n for n in source['nodeChanges']}
        for n in source['nodeChanges']:
            if n['type'] != 'TEXT':
                continue
            style = n['fontName']['style']
            assert n['fontName']['family'] == 'Inter'
            self.text_templates.setdefault(style, n)
            d = n['derivedTextData']
            self.fontmeta.setdefault(style, d['fontMetaData'][0])
            s = n['textData']['characters']
            glyphs = d.get('glyphs', [])
            starts = sorted(set(g['firstCharacter'] for g in glyphs) | {len(s)})
            counts = collections.Counter(g['firstCharacter'] for g in glyphs)
            for g in glyphs:
                i = g['firstCharacter']
                end = next((j for j in starts if j > i), len(s))
                # A cached '->' ligature is one arrow glyph, not the '-' glyph.
                # Only reuse glyphs known to cover exactly one character.
                if i < len(s) and s[i] != '\n' and end == i+1 and counts[i] == 1:
                    self.glyphs.setdefault((style, s[i]), g)
        self.checks = []
        self.quote_checks = []

    def blob(self, data):
        i = len(self.blobs)
        self.blobs.append({'bytes': {'$bytes': base64.b64encode(data).decode()}})
        return i

    def add(self, n, parent, x, y, name=None):
        n = copy.deepcopy(n)
        n['guid'] = dict(sessionID=self.session, localID=len(self.nodes)+1)
        p = gid(parent['guid'])
        position = '!' + f'{self.positions[p]:04x}'
        self.positions[p] += 1
        n['parentIndex'] = dict(guid=copy.deepcopy(parent['guid']), position=position)
        n['transform'] = transform(x, y)
        n.pop('editInfo', None)
        if name is not None:
            n['name'] = name
        self.nodes.append(n)
        return n

    def base(self, kind, name, w, h):
        return dict(phase='CREATED', type=kind, name=name, visible=True, opacity=1,
                    size=dict(x=w, y=h), strokeWeight=0, strokeAlign='CENTER', strokeJoin='MITER')

    def frame(self, parent, name, x, y, w, h, fill=None):
        n = self.base('FRAME', name, w, h)
        n['frameMaskDisabled'] = True
        n['horizontalConstraint'] = n['verticalConstraint'] = 'MIN'
        if fill:
            n['fillPaints'] = paint(fill)
            n['fillGeometry'] = [dict(windingRule='NONZERO', commandsBlob=self.blob(rect_commands(0,0,w,h)), styleID=0)]
        return self.add(n, parent, x, y)

    def rect(self, parent, name, x, y, w, h, fill=None, stroke=None, sw=0):
        n = self.base('ROUNDED_RECTANGLE', name, w, h)
        if fill:
            n['fillPaints'] = paint(fill)
            n['fillGeometry'] = [dict(windingRule='NONZERO', commandsBlob=self.blob(rect_commands(0,0,w,h)), styleID=0)]
        if stroke and sw:
            n['strokePaints'] = paint(stroke); n['strokeWeight'] = sw
            edge = rect_commands(-sw/2,-sw/2,w+sw,h+sw) + rect_commands(sw/2,sw/2,w-sw,h-sw,True)
            n['strokeGeometry'] = [dict(windingRule='NONZERO', commandsBlob=self.blob(edge), styleID=0)]
        return self.add(n, parent, x, y)

    def measure(self, s, size, bold=False):
        style = 'Bold' if bold else 'Regular'
        for c in s:
            if (style,c) not in self.glyphs:
                raise AssertionError(f'Missing reference glyph: {style}, {c!r}, text={s!r}')
        return sum(self.glyphs[style,c]['advance'] * size for c in s)

    def text(self, parent, s, x, y, size=18.3, color=INK, bold=False, align='left', maxw=None, name=None):
        style = 'Bold' if bold else 'Regular'
        width = self.measure(s, size, bold)
        height = size * self.fontmeta[style]['fontLineHeight']
        if align == 'right':
            x -= width
        elif align == 'center':
            x -= width/2
        if maxw is not None:
            assert width <= maxw, (s, width, maxw)
        n = copy.deepcopy(self.text_templates[style])
        for k in ('derivedTextData','textData','fillGeometry','strokeGeometry','editInfo','exportSettings'):
            n.pop(k, None)
        n['name'] = name or s
        n['fontSize'] = size
        n['fontName'] = copy.deepcopy(self.fontmeta[style]['key'])
        n['size'] = dict(x=math.ceil(width)+2, y=math.ceil(height)+1)
        n['textAutoResize'] = 'WIDTH_AND_HEIGHT'
        n['fillPaints'] = paint(color)
        n['strokeWeight'] = 0
        n.pop('strokePaints', None)
        n['lineHeight'] = dict(value=100, units='PERCENT')
        n['letterSpacing'] = dict(value=0, units='PIXELS')
        n['textAlignHorizontal'] = 'LEFT'
        n['textAlignVertical'] = 'TOP'
        n['textData'] = dict(characters=s, lines=[dict(lineType='PLAIN', styleId=0,
            indentationLevel=0, sourceDirectionality='AUTO', listStartOffset=0, isFirstLineOfList=False)])
        template = self.text_templates[style]
        baseline = template['derivedTextData']['baselines'][0]['position']['y'] / template['fontSize'] * size
        glyphs, offsets, advance = [], [], 0
        for i, c in enumerate(s):
            g = copy.deepcopy(self.glyphs[style,c])
            g.update(position=dict(x=advance,y=baseline), fontSize=size, firstCharacter=i, rotation=0)
            glyphs.append(g); offsets.append(advance); advance += g['advance'] * size
        n['derivedTextData'] = dict(layoutSize=n['size'], baselines=[dict(position=dict(x=0,y=baseline),
            width=width,lineY=0,lineHeight=height,lineAscent=math.ceil(baseline),firstCharacter=0,endCharacter=len(s))],
            glyphs=glyphs,fontMetaData=[copy.deepcopy(self.fontmeta[style])],truncationStartIndex=-1,
            truncatedHeight=-1,logicalIndexToCharacterOffsetMap=offsets,derivedLines=[dict(directionality='LTR')])
        n = self.add(n, parent, x, y)
        self.checks.append(dict(text=s, node=gid(n['guid']), width=width, maximum=maxw))
        return n

    def rich(self, parent, parts, x, y, size=18.3, maxw=None):
        start = x
        for s, color, bold, bg in parts:
            width = self.measure(s,size,bold)
            if bg:
                self.rect(parent, 'Highlight · '+s.strip(), x-2,y+1,width+4,size*1.25,bg)
            self.text(parent,s,x,y,size,color,bold)
            x += width
        if maxw is not None:
            assert x-start <= maxw, (parts,x-start,maxw)

    def excerpt(self, full, lines, label):
        norm = lambda s: re.sub(r'\s+','',s.replace('**','').replace('`',''))
        source, cursor = norm(full), 0
        for part in '\n'.join(lines).split('…'):
            part = norm(part)
            if part:
                found = source.find(part,cursor)
                assert found >= 0, (label,part)
                cursor = found + len(part)
        self.quote_checks.append(dict(label=label, text=lines))


def compact_outputs(snapshot, col):
    output = snapshot['outputs'][col]
    if snapshot['node'] in (2,10):
        cells = [v for v in output.split('|') if v.startswith('0,')]
        return ['|'.join(cells[i:i+2])+'|'+('…' if i+2 >= len(cells) else '') for i in range(0,len(cells),2)]
    if snapshot['node'] == 11:
        gold = re.search(r'gold:[^|]+',output).group()
        gray = re.search(r'gray:[^|]+',output).group()
        return ['… bg=black;',gold+'|',gray+'|','green:15,1|…']
    flag = re.search(r'gare=[01]',output).group()
    gold = re.search(r'gold:[^;|]+',output).group()
    gray = re.search(r'gray:[^;|]+',output).group()
    return ['… f: … '+flag+', …','|'+gold+';…','|'+gray+';…','|green:15,1;…']


def build(source,evidence):
    b = Builder(source)
    page = b.byid[0,1]
    root = b.frame(page,'grow_learning_compact',0,0,W,H,BG)
    root['frameMaskDisabled'] = False
    root['exportSettings'] = copy.deepcopy(b.byid[3,2]['exportSettings'])
    game = b.frame(root,'Game sequence',11,18,1072,231)
    notes = ['Two empty columns','One empty column','The blocks touch','Gray covers gold’s edge']
    for i, frame in enumerate(evidence['frames']):
        state = b.frame(game,f'X{i} · frame {frame["step"]}',i*286.74,0,220,231)
        # Reuse the actual imported native grid cells and their geometry, changing only position.
        cells = [n for n in source['nodeChanges'] if n['guid']['sessionID']==1 and
                 n['name'].startswith('Cell (') and n['parentIndex']['guid']['localID']==[4,281,548,815][i]]
        if len(cells) != 256:
            # Resolve frame groups by name rather than relying on export numbering.
            state_source = next(n for n in source['nodeChanges'] if n['guid']['sessionID']==1 and n['type']=='FRAME'
                                and n['name'].startswith(f'X{i} '))
            cells = [n for n in source['nodeChanges'] if n.get('parentIndex',{}).get('guid')==state_source['guid'] and n['name'].startswith('Cell (')]
        assert len(cells)==256
        cw = cells[0]['size']['x']; grid_size = 16*cw
        for n in cells:
            r,c = map(int,re.search(r'Cell \((\d+), (\d+)\)',n['name']).groups())
            assert n['name'].split(' · ')[1] == frame['grid'][r][c]
            expected = color_of(paint(frame['grid'][r][c]))
            assert all(abs(a-c)<1e-6 for a,c in zip(color_of(n['fillPaints']),expected))
            b.add(n,state,25.5+c*cw,r*cw)
        # Grid outline, labels, and the reference's bottom-right X labels.
        b.rect(state,'Grid boundary',25.5,0,grid_size,grid_size,stroke=INK,sw=1.5)
        for r in (0,2,15):
            b.text(state,str(r),18,r*cw+1,12.9,MUTED,align='right')
        for c in (0,3,8,15):
            b.text(state,str(c),25.5+(c+.5)*cw,grid_size+4,12.9,MUTED,align='center')
        b.text(state,f'X{SUBS[i]}',grid_size+37,grid_size+4,19.2,INK,True)
        b.text(state,notes[i],25.5+grid_size/2,210,16.5,INK,align='center')
    actions = b.frame(game,'Actions · left, left, left',0,0,1072,140)
    for i in range(3):
        x = 25.5+168.48+15 + i*286.74
        b.add(b.byid[3,62],actions,x,96,'Action arrow · left')
        b.add(b.byid[3,63],actions,x+77.89960479736328,91.8,'Action arrowhead · left')
        b.text(actions,'left',x+43.15,61,18.3,INK,align='center')

    headers = ['P lists colored cells','P unchanged; D learns block dynamics',
               'P groups connected cells into rectangles','P adds a predicate for gold–gray contact']
    for ri,snapshot in enumerate(evidence['snapshots']):
        it,node = snapshot['iteration'],snapshot['node']
        y = ROW_Y[ri]
        row = b.frame(root,f'Iteration {it}',18,y,1796,[151,205,151,190][ri])
        header_color = GREEN if node in (10,23) else BLUE if node==11 else MUTED
        if ri:
            b.rect(row,'Perception row separator',5,-5,1035,1,RULE)
        b.text(row,f'Iteration {it}',5,0,20.1,header_color,True)
        b.text(row,headers[ri],1044,2,18,header_color if node in (10,11,23) else MUTED,align='right')
        py = 36
        if node == 23:
            b.rich(row,[('gare',BLUE,True,BLUE_BG),(' = 1 when gold is immediately left of gray.',MUTED,False,None)],5,29,17.4,maxw=1034)
            py = 64
        for i in range(4):
            if i:
                b.rect(row,f'Separator before P(X{i})',COLS[i]-32,py+3,0.8,113,RULE)
            pg = b.frame(row,f'Perception P(X{i})',COLS[i]-18,py,210,117)
            b.text(pg,f'P(X{SUBS[i]})',0,0,17.1,MUTED,True)
            if node==10:
                b.text(pg,'unchanged',203,1,13.8,MUTED,align='right')
            lines = compact_outputs(snapshot,i)
            b.excerpt(snapshot['outputs'][i],lines,f'Iteration {it}, P(X{i})')
            for j,line in enumerate(lines):
                ly = 24+j*23
                if node==11 and j in (1,2):
                    b.rich(pg,[(line,BLUE,True,BLUE_BG)],0,ly,18.3,maxw=210)
                elif node==23 and j==0:
                    flag = 'gare='+str(int(i>=2))
                    before,after = line.split(flag)
                    b.rich(pg,[(before,MUTED,False,None),(flag,GREEN if i>=2 else BLUE,True,GREEN_BG if i>=2 else BLUE_BG),(after,MUTED,False,None)],0,ly,18.3,maxw=210)
                else:
                    b.text(pg,line,0,ly,18.3,MUTED if node in (10,23) or (node==11 and j in (0,3)) else INK,maxw=210)
        dx = BX-18
        if node==2:
            panel=b.frame(row,'World knowledge B · displayed as dynamics model D',dx,18,BW,42,GRAY_BG)
            b.text(panel,'Dynamics model D = ∅  ·  no learned rules',13,10,18.3,MUTED,maxw=BW-26)
        elif node==10:
            panel=b.frame(row,'World knowledge B · learned block rules',dx,4,BW,192,GREEN_BG)
            b.text(panel,'Dynamics model D — learned block rules (excerpts)',13,9,18.6,GREEN,True,maxw=BW-26)
            lines=['… They form two contiguous blocks …',
                   '… actions can bring them together …',
                   '… left: … If a gray block exists, it shifts left by one column …',
                   '… right: … The column vacated on the left becomes empty',
                   'unless it is immediately right of a gold block, …',
                   'in which case that column becomes gold …']
            b.excerpt(snapshot['world_knowledge'],lines,'Dynamics model learned by iteration 12')
            emphasis=['two contiguous blocks','bring them together','shifts left by one column',None,'immediately right of a gold block','that column becomes gold']
            for j,(line,mark) in enumerate(zip(lines,emphasis)):
                parts=[(line,INK,False,None)]
                if mark:
                    a,c=line.split(mark)
                    parts=[(a,INK,False,None),(mark,GREEN,True,None),(c,INK,False,None)]
                b.rich(panel,parts,16,43+j*23,18.3,maxw=BW-32)
        else:
            panel=b.frame(row,'World knowledge B · inherited from iteration 12',dx,4,BW,104 if node==23 else 82,GRAY_BG)
            b.text(panel,'Dynamics model D — unchanged from iteration 12',13,9,18.6,MUTED,True,maxw=BW-26)
            if node==11:
                b.text(panel,'… If a gray block exists, it shifts left by one column …',16,42,18.3,MUTED,maxw=BW-32)
                b.excerpt(snapshot['world_knowledge'],['… If a gray block exists, it shifts left by one column …'],f'Dynamics model at iteration {it}')
            else:
                b.text(panel,'The learned rules still describe gold and gray blocks in prose.',16,42,18.3,MUTED,maxw=BW-32)
                b.rich(panel,[('They do not reference ',MUTED,False,None),('gare',BLUE,True,BLUE_BG),('.',MUTED,False,None)],16,67,18.3,maxw=BW-32)
    b.text(root,'… marks omitted text; whitespace is reflowed.',23,1023,15.9,MUTED)
    return b,root


def render(message,root_id,png,pdf=None,svg=None):
    """Render saved native path/glyph caches, not a separately drawn approximation."""
    nodes=message['nodeChanges']; byid={gid(n['guid']):n for n in nodes}
    children=collections.defaultdict(list)
    for n in nodes:
        if 'parentIndex' in n:children[gid(n['parentIndex']['guid'])].append(n)
    paths={}
    def path(i):
        if i not in paths:paths[i]=decode_path(as_bytes(message['blobs'][i]['bytes']))
        return paths[i]
    root=byid[root_id];w,h=root['size']['x'],root['size']['y']
    fig=plt.figure(figsize=(w/144,h/144),dpi=144,facecolor=BG)
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,w);ax.set_ylim(h,0);ax.axis('off')
    def shape(p,tf,color):
        if p is not None and color is not None:
            ax.add_patch(PathPatch(p,transform=tf+ax.transData,facecolor=color,edgecolor='none',lw=0,snap=False))
    def walk(n,tf,is_root=False):
        if not n.get('visible',True):return
        ntf=tf if is_root else matrix(n)+tf
        for key,paints in [('fillGeometry','fillPaints'),('strokeGeometry','strokePaints')]:
            for geometry in n.get(key,[]):shape(path(geometry['commandsBlob']),ntf,color_of(n.get(paints)))
        if n['type']=='TEXT':
            for glyph in n['derivedTextData'].get('glyphs',[]):
                s=glyph['fontSize'];p=glyph['position']
                gt=Affine2D().scale(s,-s).translate(p['x'],p['y'])+ntf
                shape(path(glyph['commandsBlob']),gt,color_of(n.get('fillPaints')))
        for child in sorted(children[gid(n['guid'])],key=lambda c:c['parentIndex']['position']):walk(child,ntf)
    walk(root,Affine2D(),True)
    fig.savefig(png,dpi=144,facecolor=BG)
    if pdf:fig.savefig(pdf,facecolor=BG)
    if svg:fig.savefig(svg,facecolor=BG)
    plt.close(fig)


def validate(message,root_id,evidence,original=None):
    nodes=message['nodeChanges'];byid={gid(n['guid']):n for n in nodes}
    assert len(byid)==len(nodes)
    blobrefs=[]
    def scan(v,key=''):
        if isinstance(v,dict):
            for k,x in v.items():
                if k.endswith('Blob') and isinstance(x,int):blobrefs.append(x)
                else:scan(x,k)
        elif isinstance(v,list):
            for x in v:scan(x,key)
    scan(nodes)
    assert all(0<=i<len(message['blobs']) for i in blobrefs)
    for n in nodes:
        if 'parentIndex' in n:assert gid(n['parentIndex']['guid']) in byid
    descendants=[]
    def below(n):
        if gid(n['guid'])==root_id:return True
        return 'parentIndex' in n and below(byid[gid(n['parentIndex']['guid'])])
    descendants=[n for n in nodes if below(n)]
    cells=[n for n in descendants if n['name'].startswith('Cell (')]
    assert len(cells)==1024
    assert len([n for n in descendants if n['name'].startswith('Perception P(')])==16
    assert len([n for n in descendants if n['name'].startswith('World knowledge B')])==4
    for n in descendants:
        if n['type']!='TEXT':continue
        d=n['derivedTextData'];s=n['textData']['characters']
        assert len(d['glyphs'])==len(s)
        assert [g['firstCharacter'] for g in d['glyphs']]==list(range(len(s)))
        # All new nodes use translation-only transforms.
        x=y=0;c=n
        while gid(c['guid'])!=root_id:
            x+=c['transform']['m02'];y+=c['transform']['m12'];c=byid[gid(c['parentIndex']['guid'])]
        assert x>=-0.01 and y>=-0.01 and x+n['size']['x']<=W+.01 and y+n['size']['y']<=H+.01,(s,x,y)
    if original is not None:
        for n in original['nodeChanges']:assert byid[gid(n['guid'])]==n
    return dict(native_text_layers=sum(n['type']=='TEXT' for n in descendants),grid_cells=len(cells),
                perception_panels=16,belief_panels=4,all_blob_and_parent_references_valid=True,
                all_original_nodes_preserved=True if original is not None else None,figma_application_import_tested=False)


def package(reference,canvas,destination,preview,standalone):
    with zipfile.ZipFile(reference) as z:
        meta=json.loads(z.read('meta.json'))
    img=Image.open(preview).convert('RGBA');img.thumbnail((800,800))
    import io
    buf=io.BytesIO();img.save(buf,format='PNG')
    meta['file_name']='grow_learning_compact' if standalone else 'learning_evolution_with_grow_compact'
    meta['exported_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta['client_meta']['thumbnail_size']=dict(width=img.width,height=img.height)
    meta['client_meta']['render_coordinates']=dict(x=0 if standalone else 1845,y=0 if standalone else 1711,width=W,height=H)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('canvas.fig',canvas.read_bytes());z.writestr('thumbnail.png',buf.getvalue())
        z.writestr('meta.json',json.dumps(meta,separators=(',',':')))
        z.writestr('images/',b'')
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None and z.read('canvas.fig')==canvas.read_bytes()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modules',type=Path,default=Path('/tmp/grow_figma_compact/node_modules'))
    args=parser.parse_args()
    bun=shutil.which('bun') or '/home/ays57/.bun/bin/bun'
    assert (args.modules/'kiwi-schema').is_dir(),'Install kiwi-schema@0.5.0 using the setup command in this script.'
    original_hash=hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    evidence=json.loads(EVIDENCE.read_text())
    assert [s['iteration'] for s in evidence['snapshots']]==[3,12,13,27]
    snapshots=evidence['snapshots'];assert snapshots[0]['world_knowledge']==''
    candidate_path=ROOT/evidence['candidate_path']
    candidates={c['idx']:c for c in map(json.loads,candidate_path.read_text().splitlines())}
    process=list(map(json.loads,(candidate_path.parent/'process_log.jsonl').read_text().splitlines()))
    iterations={v['new_idx']:v['i'] for v in process if v.get('new_idx') is not None}
    for snapshot in snapshots:
        candidate=candidates[snapshot['node']]
        assert snapshot['perception']==candidate['perception']
        assert snapshot['world_knowledge']==candidate['world_knowledge']
        assert snapshot['iteration']==iterations[snapshot['node']]
    csv.field_size_limit(10**7)
    with (ROOT/evidence['trajectory_path']).open() as stream:
        trajectory={int(row['Step']):row for row in csv.DictReader(stream)}
    from offline_learning.validate import strip_autumn_obs_metadata
    for frame in evidence['frames']:
        row=trajectory[frame['step']]
        assert strip_autumn_obs_metadata(row['Observation'])==frame['observation']
        assert json.loads(frame['observation'])==frame['grid']
        assert row['Action']==frame['action']=='left'
    assert snapshots[1]['world_knowledge']==snapshots[2]['world_knowledge']==snapshots[3]['world_knowledge']
    assert 'gare' not in snapshots[-1]['world_knowledge']
    for snapshot in snapshots:
        for i,frame in enumerate(evidence['frames']):
            value,error=run_perceive(snapshot['perception'],frame['observation'])
            assert not error and value==snapshot['outputs'][i]
    with tempfile.TemporaryDirectory(prefix='grow-native-figma-') as td:
        tmp=Path(td)
        with zipfile.ZipFile(REFERENCE) as z:(tmp/'reference.canvas').write_bytes(z.read('canvas.fig'))
        subprocess.run([bun,str(IO),'decode',str(tmp/'reference.canvas'),str(tmp/'source.json'),'-',str(args.modules)],check=True)
        source=json.loads((tmp/'source.json').read_text())
        render(source,(3,2),OUT/'magnets_learning_compact_reference.png')
        builder,root=build(source,evidence)
        structural=[copy.deepcopy(n) for n in source['nodeChanges'] if gid(n['guid']) in [(0,0),(0,1)]]
        for n in structural:n.pop('editInfo',None)
        structural[1]['name']='Grow · compact learning evolution'
        message=dict(type='NODE_CHANGES',sessionID=0,ackID=0,nodeChangeOrder='GUID',
                     nodeChanges=structural+builder.nodes,blobs=builder.blobs)
        (tmp/'standalone.json').write_text(json.dumps(message))
        subprocess.run([bun,str(IO),'encode',str(tmp/'reference.canvas'),str(tmp/'standalone.json'),str(tmp/'grow.canvas'),str(args.modules)],check=True)
        saved=json.loads((tmp/'grow.canvas.json').read_text())
        report=validate(saved,gid(root['guid']),evidence)
        png=OUT/'grow_learning_compact.png'
        render(saved,gid(root['guid']),png,OUT/'grow_learning_compact.pdf',OUT/'grow_learning_compact.svg')
        package(REFERENCE,tmp/'grow.canvas',OUT/'grow_learning_compact.fig',png,True)
        combined=copy.deepcopy(source)
        for key in list(combined):
            if key not in ('type','sessionID','ackID','nodeChangeOrder','nodeChanges','blobs'):del combined[key]
        combined['blobs']=builder.blobs
        added=copy.deepcopy(builder.nodes);added[0]['transform']=transform(1845,1711)
        combined['nodeChanges']+=added
        (tmp/'combined.json').write_text(json.dumps(combined))
        subprocess.run([bun,str(IO),'encode',str(tmp/'reference.canvas'),str(tmp/'combined.json'),str(tmp/'combined.canvas'),str(args.modules)],check=True)
        combined_saved=json.loads((tmp/'combined.canvas.json').read_text())
        combined_report=validate(combined_saved,gid(root['guid']),evidence,source)
        package(REFERENCE,tmp/'combined.canvas',OUT/'learning_evolution_fig_with_grow_compact.fig',png,False)
        assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest()==original_hash
        report.update(reference=str(REFERENCE.relative_to(ROOT)),reference_sha256=original_hash,
                      reference_frame='magnets_learning_compact',size=dict(width=W,height=H),iterations=[3,12,13,27],
                      perception_reexecutions=16,source_checkpoints_verified=4,state_observations_verified=4,excerpt_checks=builder.quote_checks,text_width_checks=len(builder.checks),
                      source_unchanged=True,combined=combined_report,uses_figma_mcp=False,
                      dynamics_model_D='Display name for the run\'s world_knowledge (B).',
                      notes=['All text is native editable Figma text with glyph caches from the supplied Inter font.',
                             'Native path/glyph caches are rendered for previews; Figma application import is untested.',
                             'Beliefs are unchanged from iteration 12 and do not reference gare.',
                             'The selected sequence concerns cloud movement and visible sun contact, not learned conditional plant growth.'])
        (OUT/'grow_learning_compact_evidence.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('excerpt_checks','combined')},indent=2))


if __name__=='__main__':main()
