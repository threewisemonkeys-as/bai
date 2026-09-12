#!/usr/bin/env python3
"""Export the current Magnets/Grow SVGs as editable Figma-importable Sketch files.

Run: .venv/bin/python offline_learning/scripts/export_learning_evolution_figma.py
No model calls or network access. All vector geometry and text come from the
current paper SVGs. .sketch is a documented Figma import format, not a .fig file.
The exported files are checked against the official Sketch JSON schema and
round-tripped locally; this does not exercise the Figma application importer.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import uuid
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

import jsonschema
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextToPath
from fontTools.pens.recordingPen import RecordingPen
from fontTools.svgLib.path import parse_path
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / 'analysis/learning_example'
OUT = FIGURES / 'figma'
SCHEMA = json.loads((Path(__file__).parent/'figma_export/sketch-file-format.schema.json').read_text())
SCALE = 3.0  # 1188-pixel-wide frames; all dimensions, fonts and strokes scale together.
SVG = '{http://www.w3.org/2000/svg}'
TAG = 'org.learning-evolution.export'
FONTS = {}
METRICS = TextToPath()


def class_dispatched_one_of(validator, choices, instance, schema):
    """Use the schema's disjoint _class constants to avoid exponential union work."""
    if isinstance(instance, dict) and '_class' in instance:
        variants = [SCHEMA['definitions'].get(c.get('$ref', '').rsplit('/', 1)[-1], {}) for c in choices]
        classes = [v.get('properties', {}).get('_class', {}).get('const') for v in variants]
        if all(c is not None for c in classes) and len(set(classes)) == len(classes):
            if instance['_class'] in classes:
                index = classes.index(instance['_class'])
                yield from validator.descend(instance, choices[index], schema_path=index)
                return
    yield from jsonschema.Draft7Validator.VALIDATORS['oneOf'](validator, choices, instance, schema)


SchemaValidator = jsonschema.validators.extend(jsonschema.Draft7Validator, {'oneOf': class_dispatched_one_of})


def uid(key):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, TAG+'/'+key)).upper()


def color(value):
    r, g, b = to_rgb(value)
    return dict(_class='color', alpha=1.0, red=r, green=g, blue=b)


def rgb(value):
    return tuple(value[k] for k in ('red', 'green', 'blue'))


def frame(x, y, width, height):
    return dict(_class='rect', constrainProportions=False,
                x=x, y=y, width=width, height=height)


def defaults(schema):
    if '$ref' in schema:
        name = schema['$ref'].rsplit('/', 1)[-1]
        if name == 'Uuid':
            return uid('unused-default')
        if name == 'PointString':
            return '{0, 0}'
        if name == 'PointListString':
            return '{{0, 0}, {0, 0}}'
        return defaults(SCHEMA['definitions'][name])
    if 'const' in schema:
        return schema['const']
    if 'enum' in schema:
        return schema['enum'][0]
    if 'oneOf' in schema:
        return defaults(schema['oneOf'][0])
    kind = schema.get('type')
    if kind == 'object':
        return {k: defaults(schema['properties'][k]) for k in schema.get('required', [])}
    if kind == 'array':
        return [defaults(schema['items']) for _ in range(schema.get('minItems', 0))]
    if kind == 'boolean':
        return False
    if kind in ('integer', 'number'):
        return schema.get('minimum', 1 if 'exclusiveMinimum' in schema else 0)
    if kind == 'string':
        return ''
    raise ValueError(schema)


def obj(kind):
    return defaults(SCHEMA['definitions'][kind])


def layer(kind, key, name, bounds):
    result = obj(kind)
    result.update(do_objectID=uid(key), name=name, nameIsFixed=True,
                  isVisible=True, resizingConstraint=63,
                  layerListExpandedType=1, frame=frame(*bounds))
    return result


def style(key, fill=None, stroke=None, thickness=0, join='miter', cap='butt'):
    result = obj('Style')
    result.update(do_objectID=uid(key+'/style'), fills=[], borders=[], shadows=[])
    result['colorControls'].update(isEnabled=False, brightness=0, contrast=1, hue=0, saturation=1)
    result['borderOptions'].update(isEnabled=True,
        lineJoinStyle={'miter': 0, 'round': 1, 'bevel': 2}[join],
        lineCapStyle={'butt': 0, 'round': 1, 'square': 2}[cap])
    for kind, value in [('Fill', fill), ('Border', stroke)]:
        if value is None or value == 'none':
            continue
        entry = obj(kind)
        entry.update(isEnabled=True, color=color(value), fillType=0)
        entry['contextSettings']['opacity'] = 1
        entry['gradient']['stops'][0]['color'] = color(value)
        if kind == 'Fill':
            entry['patternTileScale'] = 1
            result['fills'].append(entry)
        else:
            entry.update(position=0, thickness=thickness)
            result['borders'].append(entry)
    return result


def font(family, bold):
    key = (family, bold)
    if key not in FONTS:
        path = Path(findfont(FontProperties(family=family, weight='bold' if bold else 'normal')))
        tt = TTFont(path)
        FONTS[key] = dict(path=path, ps_name=tt['name'].getDebugName(6),
                         ascent=tt['hhea'].ascent/tt['head'].unitsPerEm,
                         descent=tt['hhea'].descent/tt['head'].unitsPerEm,
                         family=family, bold=bold)
        tt.close()
    return FONTS[key]


def css(element):
    return dict(part.strip().split(':', 1) for part in element.get('style', '').split(';') if ':' in part)


def point_string(x, y):
    return '{'+f'{x:.12f}, {y:.12f}'+'}'


def point_tuple(value):
    return tuple(map(float, value.strip('{}').split(',')))


def path_layer(element, key):
    pen = RecordingPen()
    parse_path(element.attrib['d'], pen)
    points, closed = [], False
    for op, args in pen.value:
        if op in ('moveTo', 'lineTo'):
            assert op != 'moveTo' or not points, 'Compound paths need explicit subpaths'
            points.append(dict(p=args[0], incoming=None, outgoing=None))
        elif op == 'qCurveTo':
            assert len(args) == 2
            start, control, end = points[-1]['p'], args[0], args[1]
            points[-1]['outgoing'] = tuple(a+2*(b-a)/3 for a,b in zip(start,control))
            incoming = tuple(a+2*(b-a)/3 for a,b in zip(end,control))
            points.append(dict(p=end, incoming=incoming, outgoing=None))
        elif op == 'curveTo':
            points[-1]['outgoing'] = args[0]
            points.append(dict(p=args[-1], incoming=args[1], outgoing=None))
        elif op == 'closePath':
            closed = True
        elif op != 'endPath':
            raise ValueError(op)
    if closed and len(points)>1 and points[-1]['p'] == points[0]['p']:
        points[0]['incoming'] = points[-1]['incoming']
        points.pop()
    coords = [p[k] for p in points for k in ('p','incoming','outgoing') if p[k] is not None]
    left, top = min(p[0] for p in coords), min(p[1] for p in coords)
    width = max(p[0] for p in coords)-left
    height = max(p[1] for p in coords)-top
    # Avoid zero-sized native frames, without moving any vector point.
    fw, fh = max(width, .001), max(height, .001)
    corners = len(points)==4 and all(p['incoming'] is None and p['outgoing'] is None for p in points)
    rect = closed and corners and len({p['p'][0] for p in points})==2 and len({p['p'][1] for p in points})==2
    result = layer('Rectangle' if rect else 'ShapePath', key, 'Rectangle' if rect else 'Vector',
                   (left*SCALE, top*SCALE, fw*SCALE, fh*SCALE))
    result.update(isClosed=closed, points=[])
    for point in points:
        p = obj('CurvePoint')
        p.update(curveMode=4 if point['incoming'] or point['outgoing'] else 1,
                 hasCurveFrom=point['outgoing'] is not None,
                 hasCurveTo=point['incoming'] is not None)
        for prop, src in [('point','p'),('curveFrom','outgoing'),('curveTo','incoming')]:
            xx, yy = point[src] if point[src] is not None else point['p']
            p[prop] = point_string((xx-left)/fw, (yy-top)/fh)
        result['points'].append(p)
    styling = {k:v.strip() for k,v in css(element).items()}
    result['style'] = style(key, styling.get('fill', '#000000'), styling.get('stroke'),
        float(styling.get('stroke-width',1))*SCALE,
        styling.get('stroke-linejoin','round'), styling.get('stroke-linecap','butt'))
    result['userInfo'] = {TAG: dict(svg_path=element.attrib['d'], svg_id=key.rsplit('/',1)[-1])}
    return result


def text_layer(element, key):
    styling = {k:v.strip() for k,v in css(element).items()}
    family = styling['font-family'].strip("'")
    size = float(styling['font-size'].removesuffix('px'))*SCALE
    bold = styling.get('font-weight','400') in ('700','bold')
    info = font(family, bold)
    value = ''.join(element.itertext())
    assert element.get('transform', '').startswith('rotate(-0 ')
    prop = FontProperties(family=family, weight='bold' if bold else 'normal', size=size)
    width = METRICS.get_text_width_height_descent(value, prop, False)[0]
    baseline = float(element.attrib['y'])*SCALE
    left = float(element.attrib['x'])*SCALE
    anchor = styling.get('text-anchor','start')
    left -= width/2 if anchor == 'middle' else width if anchor == 'end' else 0
    height = size*(info['ascent']-info['descent'])
    top = baseline-size*info['ascent']
    result = layer('Text', key, value.strip() or 'Space', (left, top, max(.01,width), height))
    attributes = dict(
        MSAttributedStringFontAttribute=dict(_class='fontDescriptor', attributes=dict(name=info['ps_name'], size=size)),
        MSAttributedStringColorAttribute=color(styling.get('fill','#000000')),
        paragraphStyle=dict(_class='paragraphStyle', alignment=0),
        kerning=0,
    )
    result.update(textBehaviour=0, lineSpacingBehaviour=2,
        glyphBounds='{{0, 0}, {'+f'{width:.9f}, {height:.9f}'+'}}',
        attributedString=dict(_class='attributedString', string=value,
            attributes=[dict(_class='stringAttribute', location=0,
                             length=len(value.encode('utf-16-le'))//2, attributes=attributes)]))
    result['style'] = style(key)
    result['style']['textStyle'] = dict(_class='textStyle', verticalAlignment=0, encodedAttributes=deepcopy(attributes))
    result['userInfo'] = {TAG: dict(svg_id=key.rsplit('/',1)[-1], svg_baseline=baseline, svg_anchor=anchor)}
    return result


def bounds(layers):
    fs = [l['frame'] for l in layers]
    x, y = min(f['x'] for f in fs), min(f['y'] for f in fs)
    return (x, y, max(f['x']+f['width'] for f in fs)-x,
            max(f['y']+f['height'] for f in fs)-y)


def group(key, name, children):
    result = layer('Group', key, name, bounds(children))
    result['hasClickThrough'] = True
    result['groupLayout'] = obj('FreeformGroupLayout')
    for child in children:
        child['frame']['x'] -= result['frame']['x']
        child['frame']['y'] -= result['frame']['y']
    result['layers'] = children
    return result


def text_of(item):
    return item.get('attributedString',{}).get('string','')


def bucket_groups(key, items, classify):
    buckets = OrderedDict()
    for item in items:
        name = classify(item)
        buckets.setdefault(name, []).append(item)
    return [group(key+'/'+name, name, contents) for name,contents in buckets.items()]


def build_game(game):
    stem = FIGURES/f'learning_evolution_{game}_relations'
    raw = stem.with_suffix('.svg').read_bytes()
    root = ET.fromstring(raw)
    width, height = list(map(float, root.attrib['viewBox'].split()))[2:]
    original = root.find(SVG+'g')
    parents = {c:p for p in root.iter() for c in p}
    clip = root.find('.//'+SVG+'clipPath/'+SVG+'rect')
    assert clip.attrib == dict(x='0', y='0', width=str(int(width)), height=str(height))
    layers = []
    element_counts = Counter()
    for element in original.iter():
        if element.tag not in (SVG+'path', SVG+'text'):
            continue
        parent_id = parents[element].attrib['id']
        element_counts[parent_id] += 1
        key = game+'/'+parent_id+'/'+str(element_counts[parent_id])
        item = text_layer(element,key) if element.tag==SVG+'text' else path_layer(element,key)
        item['userInfo'][TAG]['svg_order'] = len(layers)
        layers.append(item)
    background = layers.pop(0)
    assert background['frame'] == frame(0,0,width*SCALE,height*SCALE)
    data = json.loads(stem.with_name(stem.name+'_evidence.json').read_text())
    wanted_iterations = [0,4,8] if game=='magnets' else [3,12,13,27]
    iteration_texts = sorted([l for l in layers if re.fullmatch(r'Iteration \d+',text_of(l))], key=lambda l:l['frame']['y'])
    assert [int(text_of(l).split()[-1]) for l in iteration_texts] == wanted_iterations
    row_separators = sorted(l['frame']['y'] for l in layers
        if l['_class']=='shapePath' and l['frame']['width']>.8*width*SCALE and l['frame']['height']<.01)
    sequence_end, notes_start = row_separators[0], row_separators[-1]
    row_starts = [iteration_texts[0]['frame']['y']-5*SCALE]
    row_starts += [max(y for y in row_separators if y<t['frame']['y']) for t in iteration_texts[1:]]
    titles = sorted([l for l in layers if re.match(r'X[₀₁₂₃]  · ',text_of(l))], key=lambda l:l['frame']['x'])
    centers = [l['frame']['x']+l['frame']['width']/2 for l in titles]
    assert len(centers)==4
    def column(item):
        f = item['frame']
        return min(range(4), key=lambda i:abs(f['x']+f['width']/2-centers[i]))
    def main_group(item):
        y = item['frame']['y']+item['frame']['height']/2
        if y<sequence_end:
            return 'Game sequence'
        if y<row_starts[0]:
            return 'Learning direction'
        if y>=notes_start:
            return 'Notes'
        return 'Iteration '+str(wanted_iterations[max(i for i,start in enumerate(row_starts) if y>=start)])
    sections = OrderedDict((n,[]) for n in ['Game sequence','Learning direction']+[f'Iteration {i}' for i in wanted_iterations]+['Notes'])
    for item in layers:
        sections[main_group(item)].append(item)
    state_cells = [0]*4
    cell = (.12*72 if game=='magnets' else .78*72/16)*SCALE
    nrows,ncols,r0,c0 = (7,6,5,4) if game=='magnets' else (16,16,0,0)
    grid_top = (.36 if game=='magnets' else .30)*72*SCALE
    def sequence_group(item):
        value, f = text_of(item), item['frame']
        if value in ('down','noop','left') or item['_class']=='shapePath':
            return 'Actions'
        if value.startswith(('Game sequence','Gameplay')):
            return 'Sequence label'
        i = column(item)
        if item['_class']=='rectangle' and abs(f['width']-cell)<.001 and abs(f['height']-cell)<.001:
            c = round((f['x']-(centers[i]-ncols*cell/2))/cell)+c0
            r = round((f['y']-grid_top)/cell)+r0
            actual = data['frames'][i]['grid'][r][c]
            assert rgb(item['style']['fills'][0]['color']) == to_rgb(actual)
            item['name'] = f'Cell ({r}, {c}) · {actual}'
            state_cells[i] += 1
        return f'X{i} · frame {data["steps"][i]}'
    result_groups = []
    for name, items in sections.items():
        if name=='Game sequence':
            children = bucket_groups(game+'/'+name,items,sequence_group)
        elif name.startswith('Iteration'):
            py = min(l['frame']['y'] for l in items if text_of(l).startswith('P(X'))
            panels = [l for l in items if l['_class']=='rectangle' and l['frame']['width']>.8*width*SCALE]
            by = min((l['frame']['y'] for l in panels), default=float('inf'))
            def row_group(item):
                y = item['frame']['y']+item['frame']['height']/2
                if y>=by or text_of(item).startswith(('World knowledge B','B =','B —')):
                    return 'World knowledge B'
                if y<py:
                    return 'Checkpoint and explanation'
                return f'Perception P(X{column(item)})'
            children = bucket_groups(game+'/'+name,items,row_group)
        else:
            children = items
        result_groups.append(group(game+'/'+name,name,children))
    assert state_cells == [nrows*ncols]*4, state_cells
    artboard = layer('Artboard',game+'/artboard',game.title()+' · learning evolution',(0,0,width*SCALE,height*SCALE))
    artboard.update(layers=result_groups, hasBackgroundColor=True, includeBackgroundColorInExport=True,
                    backgroundColor=background['style']['fills'][0]['color'], resizesContent=False)
    artboard['userInfo'] = {TAG: dict(source=stem.with_suffix('.svg').name,
        source_sha256=hashlib.sha256(raw).hexdigest(), scale=SCALE, iterations=wanted_iterations,
        provenance=stem.with_name(stem.name+'_evidence.json').name)}
    page = layer('Page',game+'/page',game.title(),(0,0,width*SCALE,height*SCALE))
    page.update(layers=[artboard], hasClickThrough=True)
    stats = dict(game=game, iterations=wanted_iterations, text_layers=sum(l['_class']=='text' for l in layers),
                 vector_layers=sum(l['_class'] in ('rectangle','shapePath') for l in layers),
                 grid_cells=state_cells, frame_width=width*SCALE, frame_height=height*SCALE,
                 source_sha256=hashlib.sha256(raw).hexdigest())
    verify_native_page(page, game)
    return page,stats


def document(pages):
    doc = defaults(SCHEMA['properties']['document'])
    doc.update(do_objectID=uid('document/'+','.join(p['name'] for p in pages)), pages=pages, colorSpace=1)
    doc['assets']['do_objectID'] = uid('assets')
    meta = obj('Meta')
    # Target serialization format, not a claim that the Sketch application ran.
    meta.update(commit='generated-from-paper-svg', version=146, appVersion='101.0', build=0,
        pagesAndArtboards={p['do_objectID']:dict(name=p['name'],artboards={a['do_objectID']:dict(name=a['name']) for a in p['layers']}) for p in pages})
    meta['created'].update(commit=meta['commit'],version=meta['version'],appVersion=meta['appVersion'],build=0,compatibilityVersion=99)
    user = obj('User')
    user['document'].update(pageListHeight=120,pageListCollapsed=0)
    for p in pages:
        user[p['do_objectID']] = dict(scrollOrigin='{0, 0}', zoomValue=.6)
    expanded = dict(document=doc,meta=meta,user=user)
    SchemaValidator(SCHEMA).validate(expanded)
    return expanded


def write_sketch(path, pages, preview):
    expanded = document(pages)
    doc = deepcopy(expanded['document'])
    doc['pages'] = [dict(_class='MSJSONFileReference', _ref_class='MSImmutablePage', _ref='pages/'+p['do_objectID']) for p in pages]
    with ZipFile(path,'w',ZIP_DEFLATED) as archive:
        for name,value in [('document',doc),('meta',expanded['meta']),('user',expanded['user'])]:
            archive.writestr(name+'.json',json.dumps(value,ensure_ascii=False,separators=(',',':')))
        for page in pages:
            archive.writestr('pages/'+page['do_objectID']+'.json',json.dumps(page,ensure_ascii=False,separators=(',',':')))
        archive.writestr('previews/preview.png',preview)
    # Re-open the actual file, resolve every page reference, and re-validate it.
    with ZipFile(path) as archive:
        assert archive.testzip() is None
        check_doc = json.loads(archive.read('document.json'))
        check_doc['pages'] = [json.loads(archive.read(p['_ref']+'.json')) for p in check_doc['pages']]
        check = dict(document=check_doc,meta=json.loads(archive.read('meta.json')),user=json.loads(archive.read('user.json')))
        SchemaValidator(SCHEMA).validate(check)
        assert check == expanded


def walk(layers,x=0,y=0):
    for item in layers:
        f = item['frame']
        px,py = x+f['x'], y+f['y']
        if 'layers' in item:
            yield from walk(item['layers'],px,py)
        else:
            yield item,px,py


def render_native(page):
    """Read native layer coordinates/styles to render a local geometry preview."""
    board = page['layers'][0]
    width,height = board['frame']['width'],board['frame']['height']
    output_width = 1650
    ratio = output_width/width
    fig = plt.figure(figsize=(output_width/100,height*ratio/100),dpi=100,facecolor=rgb(board['backgroundColor']))
    ax = fig.add_axes([0,0,1,1],xlim=(0,width),ylim=(height,0))
    ax.set_axis_off()
    for item,x,y in walk(board['layers']):
        styling = item['style']
        if item['_class']=='text':
            attr = item['attributedString']['attributes'][0]['attributes']
            name,size = (attr['MSAttributedStringFontAttribute']['attributes'][k] for k in ('name','size'))
            info = next(f for f in FONTS.values() if f['ps_name']==name)
            baseline = y+size*info['ascent']
            ax.text(x,baseline,item['attributedString']['string'],fontsize=size*ratio*72/100,
                    family=info['family'],weight='bold' if info['bold'] else 'normal',
                    color=rgb(attr['MSAttributedStringColorAttribute']),va='baseline',ha='left',zorder=2)
        else:
            w,h = item['frame']['width'],item['frame']['height']
            def xy(point,key):
                a,b = point_tuple(point[key])
                return x+a*w,y+b*h
            points = item['points']
            vertices,codes = [xy(points[0],'point')],[MplPath.MOVETO]
            segments = list(zip(points,points[1:]))
            if item['isClosed']:
                segments.append((points[-1],points[0]))
            for start,end in segments:
                if start['hasCurveFrom'] or end['hasCurveTo']:
                    vertices.extend([xy(start,'curveFrom'),xy(end,'curveTo'),xy(end,'point')])
                    codes.extend([MplPath.CURVE4]*3)
                else:
                    vertices.append(xy(end,'point')); codes.append(MplPath.LINETO)
            if item['isClosed']:
                vertices.append(vertices[0]); codes.append(MplPath.CLOSEPOLY)
            fills,borders = styling['fills'],styling['borders']
            ax.add_patch(PathPatch(MplPath(vertices,codes),
                facecolor=rgb(fills[0]['color']) if fills else 'none',
                edgecolor=rgb(borders[0]['color']) if borders else 'none',
                lw=borders[0]['thickness']*ratio*72/100 if borders else 0,
                joinstyle=['miter','round','bevel'][styling['borderOptions']['lineJoinStyle']],
                capstyle=['butt','round','projecting'][styling['borderOptions']['lineCapStyle']],zorder=1))
    buffer = io.BytesIO()
    fig.savefig(buffer,format='png',dpi=100,facecolor=fig.get_facecolor())
    plt.close(fig)
    return buffer.getvalue()


def verify_native_page(page, game):
    source = ET.parse(FIGURES/f'learning_evolution_{game}_relations.svg').getroot()
    board = page['layers'][0]
    native = list(walk(board['layers']))
    expected_text = Counter(''.join(t.itertext()) for t in source.iter(SVG+'text'))
    actual_text = Counter(text_of(item) for item,x,y in native if item['_class']=='text')
    assert actual_text == expected_text
    assert len(native) == len(list(source.iter(SVG+'path')))+len(list(source.iter(SVG+'text')))-1
    ids=[]
    def gather(value):
        if isinstance(value,dict):
            if 'do_objectID' in value:
                ids.append(value['do_objectID'])
            for child in value.values():
                gather(child)
        elif isinstance(value,list):
            for child in value:
                gather(child)
    gather(page)
    assert len(ids)==len(set(ids)), 'Native layer/style IDs must be unique'
    for item,x,y in native:
        assert min(x,y)>=-.02, (item['name'],x,y)
        assert x+item['frame']['width']<=board['frame']['width']+.02, item['name']
        assert y+item['frame']['height']<=board['frame']['height']+.02, item['name']
    for row in board['layers']:
        if row['name'].startswith('Iteration '):
            assert {g['name'] for g in row['layers'] if g['name'].startswith('Perception')} == {f'Perception P(X{i})' for i in range(4)}
            assert any(g['name']=='World knowledge B' for g in row['layers'])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pages,stats,previews = [],[],[]
    for game in ('magnets','grow'):
        page,summary = build_game(game)
        preview = render_native(page)
        path = OUT/f'learning_evolution_{game}_relations.sketch'
        write_sketch(path,[page],preview)
        (OUT/f'{game}_editable_preview.png').write_bytes(preview)
        shutil.copy2(FIGURES/f'learning_evolution_{game}_relations.svg',OUT/f'learning_evolution_{game}_relations.svg')
        pages.append(page); stats.append(summary); previews.append(preview)
        print(game+':',json.dumps(summary),flush=True)
    write_sketch(OUT/'learning_evolution_magnets_and_grow.sketch',pages,previews[0])
    (OUT/'fonts').mkdir(exist_ok=True)
    for info in FONTS.values():
        shutil.copy2(info['path'],OUT/'fonts'/info['path'].name)
    shutil.copy2(next(iter(FONTS.values()))['path'].parent/'LICENSE_DEJAVU',OUT/'fonts/LICENSE_DEJAVU.txt')
    (OUT/'validation.json').write_text(json.dumps(dict(
        format='Sketch 146, official schema @sketch-hq/sketch-file-format 6.5.0',
        figma_import_tested=False,checks=['Official JSON schema validation','Archive integrity and page-reference round trip',
            'Every text fragment retained as a native text layer','Every game grid cell verified against source observations',
            'Native geometry preview rendered from exported layer coordinates/styles'],
        figures=stats),indent=2)+'\n')
    (OUT/'README.md').write_text('''# Editable Magnets and Grow diagrams for Figma

Import **learning_evolution_magnets_and_grow.sketch** for both diagrams on separate
pages, or import the individual `.sketch` files. These use Figma's supported
Sketch import format; they are not native `.fig` files or published Figma links.
No Sketch application or custom Figma plugin is required.

1. Install the four fonts in `fonts/` if they are not already available in Figma.
2. In Figma's file browser, choose **Create → Import → From your computer** and
   select the combined or individual `.sketch` file. Dragging it onto the file
   browser also works.
3. Open the imported design. Each page contains an editable diagram frame.

The layers are grouped into the game sequence, individual states, action arrows,
and iterations. Each iteration has separate perception groups for X0–X3 and a
world knowledge group. Grid cells, arrows, highlights, and text remain editable.
Colored text runs remain separate text layers to retain their exact positioning.
The frames are 1188 px wide (the paper SVG geometry scaled uniformly by 3).

Magnets retains iterations 0, 4, and 8, with relation definitions in iteration 8.
Grow retains iterations 3, 12, 13, and 27. Both retain the latest formatting edits.
Source programs, beliefs, and full-output evidence remain in the parent figures
directory; their content has not been changed for this export.

The DejaVu fonts and their license are included. Figma's desktop app can use
installed fonts; browser users may need Figma's font installer. See the official
import instructions for platform-specific font support.

The SVG files are alternate vector imports. Use the `.sketch` files when editable
text is important. PNG previews were rendered locally from the exported native
layers. Archive structure, the official JSON schema, exact text preservation,
and every displayed grid cell were checked. The Figma application importer was
not run in this session because a Figma connection was not available.

Regenerate after updating the paper SVGs:

    .venv/bin/python offline_learning/scripts/export_learning_evolution_figma.py

Official documentation:
- [Figma: Import Sketch files](https://help.figma.com/hc/en-us/articles/360040514273-Import-Sketch-files)
- [Sketch document format](https://developer.sketch.com/file-format/)
- [Official JSON schemas](https://github.com/sketch-hq/sketch-document)
''')
    zip_path = FIGURES/'learning_evolution_figma.zip'
    names = ['README.md','validation.json','learning_evolution_magnets_and_grow.sketch']
    names += [f'learning_evolution_{game}_relations.{ext}' for game in ('magnets','grow') for ext in ('sketch','svg')]
    names += [f'{game}_editable_preview.png' for game in ('magnets','grow')]
    names += [str(p.relative_to(OUT)) for p in sorted((OUT/'fonts').iterdir())]
    with ZipFile(zip_path,'w',ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(OUT/name,'learning_evolution_figma/'+name)
    print('Wrote',zip_path.relative_to(ROOT))


if __name__=='__main__':
    main()
