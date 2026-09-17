#!/usr/bin/env python3
"""Add three editable REx tree insets to copies of magnets_learning_compact.

Uses the original document's embedded Kiwi schema and Inter glyph caches.
Extra glyphs come from matching Inter 3.19 fonts, obtained from rsms/inter.
The original frame is preserved; three named variants are appended to the
same .fig file. Outputs include native-geometry previews and a provenance report.
"""
from __future__ import annotations

import argparse
import base64
import collections
import copy
import datetime
import hashlib
import io
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from offline_learning.scripts.fig_grow_learning_compact import (
    BG, BLUE, BLUE_BG, GREEN, GREEN_BG, INK, MUTED, RULE,
    Builder, gid, matrix, paint, render, transform,
)
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

SOURCE = ROOT / 'analysis/learning_example/learning_evolution_fig.fig'
OUT = SOURCE.parent / 'rex_tree_search'
EVIDENCE = SOURCE.parent / 'learning_evolution_magnets_relations_evidence.json'
IO = ROOT / 'offline_learning/scripts/figma_export/fig_kiwi_io.cjs'
PREFIX = 'magnets_learning_compact_rex_'
VARIANTS = ['growth', 'lineages', 'selection']
TITLES = ['Tree growth', 'Alternative lineages', 'Explore / exploit']
IX, IY, IW, IH = 1097, 18, 717, 266
GRAY, LIGHT = '#a8b0b6', '#eef0f1'


def command(tag, *coords):
    return bytes([tag]) + struct.pack('<' + 'f' * len(coords), *coords)


class GlyphPen(BasePen):
    def __init__(self, glyphset, units):
        super().__init__(glyphset)
        self.units, self.commands = units, []

    def point(self, p):
        return tuple(v / self.units for v in p)

    def _moveTo(self, p):
        self.commands.append(command(1, *self.point(p)))

    def _lineTo(self, p):
        self.commands.append(command(2, *self.point(p)))

    def _curveToOne(self, a, b, c):
        self.commands.append(command(4, *self.point(a), *self.point(b), *self.point(c)))

    def _qCurveToOne(self, a, b):
        self.commands.append(command(3, *self.point(a), *self.point(b)))

    def _closePath(self):
        self.commands.append(command(0))

    def _endPath(self):
        pass


def circle_path(cx, cy, radius, reverse=False):
    k = 4 * (math.sqrt(2) - 1) / 3
    direction = -1 if reverse else 1
    result = command(1, cx + radius, cy)
    for i in range(4):
        a, z = direction * i * math.pi / 2, direction * (i + 1) * math.pi / 2
        p = (cx + radius * math.cos(a), cy + radius * math.sin(a))
        q = (cx + radius * math.cos(z), cy + radius * math.sin(z))
        c1 = (p[0] - direction * k * radius * math.sin(a),
              p[1] + direction * k * radius * math.cos(a))
        c2 = (q[0] + direction * k * radius * math.sin(z),
              q[1] - direction * k * radius * math.cos(z))
        result += command(4, *c1, *c2, *q)
    return result + command(0)


class TreeBuilder(Builder):
    def __init__(self, source, font_dir):
        super().__init__(source)
        self.font_dir, self.fonts = font_dir, {}
        self.extra_glyphs = []
        self.edges = []

    def measure(self, s, size, bold=False):
        style = 'Bold' if bold else 'Regular'
        missing = set(s) - {c for st, c in self.glyphs if st == style}
        if missing:
            if style not in self.fonts:
                path = self.font_dir / f'Inter-{style}.woff'
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    url = f'https://raw.githubusercontent.com/rsms/inter/v3.19/docs/font-files/Inter-{style}.woff'
                    path.write_bytes(urlopen(url, timeout=30).read())
                font = TTFont(path)
                assert font['head'].unitsPerEm == 2816
                self.fonts[style] = font
            font = self.fonts[style]
            glyphset, cmap = font.getGlyphSet(), font.getBestCmap()
            for c in sorted(missing):
                name = cmap[ord(c)]
                pen = GlyphPen(glyphset, font['head'].unitsPerEm)
                glyphset[name].draw(pen)
                self.glyphs[style, c] = dict(
                    commandsBlob=self.blob(b''.join(pen.commands)),
                    position=dict(x=0, y=0), fontSize=1, firstCharacter=0,
                    advance=font['hmtx'][name][0] / font['head'].unitsPerEm,
                    rotation=0,
                )
                self.extra_glyphs.append([style, c])
        return super().measure(s, size, bold)

    def circle(self, parent, name, cx, cy, radius, fill, stroke=None, sw=1.6):
        n = self.base('ROUNDED_RECTANGLE', name, 2 * radius, 2 * radius)
        n['cornerRadius'] = radius
        if fill:
            n['fillPaints'] = paint(fill)
            n['fillGeometry'] = [dict(windingRule='NONZERO', styleID=0,
                commandsBlob=self.blob(circle_path(radius, radius, radius)))]
        if stroke:
            n['strokeWeight'], n['strokePaints'] = sw, paint(stroke)
            ring = (circle_path(radius, radius, radius + sw / 2) +
                    circle_path(radius, radius, radius - sw / 2, True))
            n['strokeGeometry'] = [dict(windingRule='NONZERO', styleID=0,
                commandsBlob=self.blob(ring))]
        return self.add(n, parent, cx - radius, cy - radius)

    def line(self, parent, name, p, q, color=GRAY, width=1.7, dashed=False):
        dx, dy = q[0] - p[0], q[1] - p[1]
        length = math.hypot(dx, dy)
        if not length:
            return
        ux, uy = dx / length, dy / length
        if dashed:
            for i, start in enumerate(range(0, math.ceil(length), 9)):
                stop = min(start + 5, length)
                if stop > start:
                    self.line(parent, f'{name} / dash {i}',
                        (p[0] + ux * start, p[1] + uy * start),
                        (p[0] + ux * stop, p[1] + uy * stop), color, width)
            return
        n = self.rect(parent, name, 0, 0, length, width, color)
        n['transform'] = dict(m00=ux, m01=-uy, m02=p[0] + uy * width / 2,
                              m10=uy, m11=ux, m12=p[1] - ux * width / 2)

    def arrow(self, parent, name, p, q, color=GRAY, width=1.7, head=5):
        self.line(parent, name, p, q, color, width)
        a = math.atan2(q[1] - p[1], q[0] - p[0])
        for sign in [-1, 1]:
            z = a + math.pi + sign * math.pi / 5
            self.line(parent, name + ' / arrowhead', q,
                      (q[0] + head * math.cos(z), q[1] + head * math.sin(z)), color, width)

    def edge(self, parent, a, z, positions, color=GRAY, width=1.7, dashed=False, radius=13.5):
        p, q = positions[a], positions[z]
        dx, dy = q[0] - p[0], q[1] - p[1]
        length = math.hypot(dx, dy)
        p = (p[0] + radius * dx / length, p[1] + radius * dy / length)
        q = (q[0] - radius * dx / length, q[1] - radius * dy / length)
        self.line(parent, f'Refinement {a} to {z}', p, q, color, width, dashed)
        self.edges.append(dict(inset=gid(parent['guid']), parent=a, child=z, dashed=dashed))

    def node(self, parent, idx, p, color=GRAY, fill=BG, radius=13.5, selected=False):
        if selected:
            self.circle(parent, f'Candidate {idx} / selected parent ring', *p, radius + 5, None, color, 1.3)
        self.circle(parent, f'Candidate {idx} / model (P, D)', *p, radius, fill, color, 1.8)
        self.text(parent, str(idx), p[0], p[1] - 9.4, 15.4,
                  '#ffffff' if fill in [BLUE, GREEN, INK] else color,
                  True, align='center', name=f'Candidate {idx} / iteration number')

    def inset(self, root, name, title):
        n = self.frame(root, 'REx inset / ' + name, IX, IY, IW, IH, BG)
        self.text(n, title, 13, 0, 20, INK, True, maxw=IW - 26)
        self.line(n, 'Title rule', (13, 31), (IW - 13, 31), RULE, 1)
        return n


def source_tree():
    e = json.loads(EVIDENCE.read_text())
    candidates = {r['idx']: r for r in map(json.loads, (ROOT / e['candidate_path']).read_text().splitlines())}
    process = [r for r in map(json.loads, (ROOT / e['process_path']).read_text().splitlines()) if r['i'] <= 8]
    assert [r['i'] for r in process] == list(range(1, 9))
    edges = [(r['selected'], r['new_idx']) for r in process]
    for r in process:
        assert r['parent_ids'] == candidates[r['new_idx']]['parents'] == [r['selected']]
        assert r['new_idx'] == r['i']
        assert math.isclose(r['new_score'], candidates[r['new_idx']]['train_score'])
    assert edges == [(0, 1), (0, 2), (0, 3), (3, 4), (1, 5), (1, 6), (4, 7), (6, 8)]
    pulls = collections.Counter(r['selected'] for r in process if r['i'] < 8)
    assert pulls[4] == 1 and pulls[6] == 0
    assert e['lineages'] == {'0': [0], '4': [0, 3, 4], '8': [0, 1, 6, 8]}
    return e, candidates, process, edges, pulls


def growth(b, root, edges):
    n = b.inset(root, '01 tree growth', 'REx search · expanding tree')
    relative = {0: (0, 0), 1: (-55, 42), 2: (0, 42), 3: (55, 42),
                4: (55, 84), 5: (-82, 84), 6: (-28, 84), 7: (55, 126), 8: (-28, 126)}
    for j, (iteration, cx) in enumerate([(0, 110), (4, 349), (8, 588)]):
        color = [MUTED, GREEN, BLUE][j]
        b.text(n, f'Iteration {iteration}', cx, 42, 17, color, True, align='center')
        pos = {i: (cx + p[0], 83 + p[1]) for i, p in relative.items()}
        last = [0, 0, 4][j]
        for a, z in edges:
            if z <= iteration:
                b.edge(n, a, z, pos, color if z > last else GRAY, 2.0 if z > last else 1.5)
        for i in range(iteration + 1):
            fill = GREEN if i == 4 else BLUE if i == 8 else BG
            c = GREEN if i == 4 else BLUE if i == 8 else color if i > last else MUTED
            b.node(n, i, pos[i], c, fill)
        b.text(n, f'{iteration + 1} candidate' + ('s' if iteration else ''), cx, 224,
               14.7, MUTED, align='center')
    b.arrow(n, 'Search grows to iteration 4', (210, 132), (242, 132), GRAY, 1.8, 6)
    b.arrow(n, 'Search grows to iteration 8', (449, 132), (481, 132), GRAY, 1.8, 6)
    b.text(n, 'Nodes: (P, D) candidates · colored edges: new refinements', 13, 247,
           14.7, MUTED, maxw=IW - 26)
    return n


def lineages(b, root, edges):
    n = b.inset(root, '02 alternative lineages', 'REx search · alternative lineages')
    pos = {0: (34, 139), 3: (144, 84), 4: (265, 84), 7: (373, 120),
           2: (144, 137), 1: (144, 193), 5: (265, 222), 6: (265, 178), 8: (373, 178)}
    green, blue = {(0, 3), (3, 4)}, {(0, 1), (1, 6), (6, 8)}
    for a, z in edges:
        c = GREEN if (a, z) in green else BLUE if (a, z) in blue else GRAY
        b.edge(n, a, z, pos, c, 2.6 if c != GRAY else 1.5)
    b.line(n, 'Iteration 4 annotation leader', (284, 84), (422, 84), GREEN, 1.1, dashed=True)
    b.line(n, 'Iteration 8 annotation leader', (393, 178), (422, 178), BLUE, 1.1)
    for i, p in pos.items():
        c = GREEN if i in [3, 4] else BLUE if i in [1, 6, 8] else MUTED
        fill = GREEN if i == 4 else BLUE if i == 8 else BG
        b.node(n, i, p, c, fill)
    b.text(n, 'Iteration 4 · coordinates', 436, 62, 18.3, GREEN, True, maxw=268)
    b.text(n, '0 → 3 → 4', 436, 91, 17, MUTED)
    b.text(n, 'Iteration 8 · relations', 436, 156, 18.3, BLUE, True, maxw=268)
    b.text(n, '0 → 1 → 6 → 8', 436, 185, 17, MUTED)
    b.text(n, 'Each node is a candidate (P, D); highlighted paths lead to the models below.',
           13, 247, 14.7, MUTED, maxw=IW - 26)
    return n


def selection(b, root, edges, candidates, pulls):
    n = b.inset(root, '03 explore exploit selection', 'REx search · explore / exploit')
    b.text(n, 'Iteration 8', 704, 4, 16, MUTED, align='right')
    pos = {0: (32, 132), 1: (129, 84), 2: (129, 143), 3: (129, 211),
           4: (232, 211), 5: (232, 56), 6: (232, 116), 7: (331, 227), 8: (331, 116)}
    for a, z in edges:
        b.edge(n, a, z, pos, BLUE if z == 8 else GRAY,
               2.6 if z == 8 else 1.5, dashed=z == 8,
               radius=19 if a == 6 else 13.5)
    for i, p in pos.items():
        c = GREEN if i == 4 else BLUE if i in [6, 8] else MUTED
        fill = GREEN_BG if i == 4 else BLUE if i == 8 else BG
        b.node(n, i, p, c, fill, selected=i == 6)
    b.text(n, 'refine', 281, 87, 14.7, BLUE, align='center')
    b.text(n, 'new', 331, 138, 14.7, BLUE, align='center')
    x = 388
    for idx, y, color, title in [(4, 49, GREEN, 'Exploit · higher score'), (6, 125, BLUE, 'Explore · less tried')]:
        b.text(n, title, x, y, 18.3, color, True, maxw=IW - x - 13)
        h, count = candidates[idx]['train_score'], pulls[idx]
        b.text(n, f'Candidate {idx}: score {h:.2f}, expanded {count}'+(' time' if count == 1 else ' times'),
               x, y + 25, 14.7, MUTED, maxw=IW - x - 13)
        b.rect(n, f'Candidate {idx} score bar background', x, y + 50, 287, 5, LIGHT)
        b.rect(n, f'Candidate {idx} training score {h:.8f}', x, y + 50, 287 * h, 5, color)
    b.text(n, 'Thompson sampling selects 6.', x, 201, 15.8, INK, maxw=IW - x - 13)
    b.text(n, 'Refinement adds candidate 8.', x, 224, 15.8, INK, maxw=IW - x - 13)
    b.text(n, 'Solid: existing refinements · dashed: new refinement · ring: selected parent',
           13, 247, 14.7, MUTED, maxw=IW - 26)
    return n


def child_map(source):
    children = collections.defaultdict(list)
    for n in source['nodeChanges']:
        if 'parentIndex' in n:
            children[gid(n['parentIndex']['guid'])].append(n)
    for values in children.values():
        values.sort(key=lambda n: n['parentIndex']['position'])
    return children


def without_previous_variants(source):
    children = child_map(source)
    remove = set()
    def mark(n):
        remove.add(gid(n['guid']))
        for child in children[gid(n['guid'])]:
            mark(child)
    for n in source['nodeChanges']:
        if n.get('name', '').startswith(PREFIX):
            mark(n)
    result = copy.deepcopy(source)
    result['nodeChanges'] = [n for n in result['nodeChanges'] if gid(n['guid']) not in remove]
    return result


def clone_frame(b, source, original, page, name, x, y):
    children = child_map(source)
    mapping = {}
    def clone(n, parent, is_root=False):
        t = n['transform']
        c = b.add(n, parent, x if is_root else t['m02'], y if is_root else t['m12'], name if is_root else n['name'])
        if not is_root:
            c['transform'] = copy.deepcopy(t)
        mapping[gid(n['guid'])] = gid(c['guid'])
        for child in children[gid(n['guid'])]:
            clone(child, c)
        return c
    root = clone(original, page, True)
    return root, mapping


def validate(saved, source, b, roots, insets, clone_maps, edges):
    byid = {gid(n['guid']): n for n in saved['nodeChanges']}
    assert len(byid) == len(saved['nodeChanges'])
    for n in source['nodeChanges']:
        assert byid[gid(n['guid'])] == n, ('Original node changed', n['name'])
    blobrefs = []
    def scan(v):
        if isinstance(v, dict):
            for k, x in v.items():
                if k.endswith('Blob') and isinstance(x, int):
                    blobrefs.append(x)
                else:
                    scan(x)
        elif isinstance(v, list):
            for x in v:
                scan(x)
    scan(saved['nodeChanges'])
    assert all(0 <= i < len(saved['blobs']) for i in blobrefs)
    for n in saved['nodeChanges']:
        if 'parentIndex' in n:
            assert gid(n['parentIndex']['guid']) in byid
    inset_ids = {gid(n['guid']) for n in insets}
    text_count, bounded = 0, 0
    for n in b.nodes:
        current, ancestors = n, []
        while gid(current['guid']) not in inset_ids and 'parentIndex' in current:
            ancestors.append(current)
            current = byid[gid(current['parentIndex']['guid'])]
        if gid(current['guid']) not in inset_ids:
            continue
        if n['type'] == 'TEXT':
            text_count += 1
            d, s = n['derivedTextData'], n['textData']['characters']
            assert len(d['glyphs']) == len(s)
            assert [g['firstCharacter'] for g in d['glyphs']] == list(range(len(s)))
        if ancestors:
            from matplotlib.transforms import Affine2D
            tf = Affine2D()
            for a in ancestors:
                tf += matrix(a)
            w, h = n['size']['x'], n['size']['y']
            points = tf.transform([(0, 0), (w, 0), (w, h), (0, h)])
            assert min(p[0] for p in points) >= -.01, (n['name'], points)
            assert max(p[0] for p in points) <= IW + .01, (n['name'], points)
            assert min(p[1] for p in points) >= -.01, (n['name'], points)
            assert max(p[1] for p in points) <= IH + .01, (n['name'], points)
            bounded += 1
    for record in b.edges:
        assert (record['parent'], record['child']) in edges
    for inset in insets[1:]:
        actual = [(r['parent'], r['child']) for r in b.edges if r['inset'] == gid(inset['guid'])]
        assert actual == edges
    first = [(r['parent'], r['child']) for r in b.edges if r['inset'] == gid(insets[0]['guid'])]
    assert first == edges[:4] + edges
    for mapping in clone_maps:
        for old, new in mapping.items():
            a, z = b.byid[old], byid[new]
            for key in ['size', 'fillGeometry', 'strokeGeometry', 'fillPaints', 'strokePaints', 'textData', 'derivedTextData', 'vectorData']:
                assert a.get(key) == z.get(key), (old, new, key)
    return dict(original_nodes_preserved=len(source['nodeChanges']),
                variants=len(roots), native_inset_text_layers=text_count,
                inset_elements_within_bounds=bounded, valid_blob_references=len(blobrefs),
                recorded_search_edges_verified=True, source_frame_geometry_preserved_in_all_variants=True,
                figma_application_import_tested=False)


def package(source_path, canvas, destination, preview, roots):
    img = Image.open(preview).convert('RGBA')
    img.thumbnail((800, 800))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    with zipfile.ZipFile(source_path) as original:
        meta = json.loads(original.read('meta.json'))
        meta['exported_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta['client_meta']['thumbnail_size'] = dict(width=img.width, height=img.height)
        meta['client_meta']['render_coordinates'] = dict(x=roots[0]['transform']['m02'],
            y=roots[0]['transform']['m12'], width=1829, height=725)
        with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_STORED) as z:
            for item in original.infolist():
                if item.filename not in ['canvas.fig', 'meta.json', 'thumbnail.png']:
                    z.writestr(item, original.read(item.filename))
            z.writestr('canvas.fig', canvas.read_bytes())
            z.writestr('thumbnail.png', buffer.getvalue())
            z.writestr('meta.json', json.dumps(meta, separators=(',', ':')))
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        assert z.read('canvas.fig') == canvas.read_bytes()


def comparison():
    # Contact sheet made only from the rendered native figures, with editorial labels.
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
    canvas = Image.new('RGB', (769, 3 * 326 + 22), BG)
    draw = ImageDraw.Draw(canvas)
    for i, (name, title) in enumerate(zip(VARIANTS, TITLES)):
        y = 17 + 326 * i
        draw.text((26, y), f'{i + 1}. {title}', fill=INK, font=font)
        inset = Image.open(OUT / f'rex_{name}_inset.png').convert('RGB')
        canvas.paste(inset, (26, y + 35))
    canvas.save(OUT / 'rex_three_options.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--modules', type=Path, default=Path('/tmp/grow_figma_compact/node_modules'))
    parser.add_argument('--font-dir', type=Path, default=Path('/tmp/rex_magnets_figma'))
    parser.add_argument('--apply', action='store_true', help='Replace the specified local .fig after validation, retaining a backup.')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bun = shutil.which('bun') or '/home/ays57/.bun/bin/bun'
    assert (args.modules / 'kiwi-schema').is_dir()
    before = hashlib.sha256(args.source.read_bytes()).hexdigest()
    evidence, candidates, process, edges, pulls = source_tree()
    with tempfile.TemporaryDirectory(prefix='magnets-rex-') as directory:
        tmp = Path(directory)
        with zipfile.ZipFile(args.source) as z:
            (tmp / 'reference.canvas').write_bytes(z.read('canvas.fig'))
        subprocess.run([bun, str(IO), 'decode', str(tmp / 'reference.canvas'), str(tmp / 'source.json'), '-', str(args.modules)], check=True)
        source = without_previous_variants(json.loads((tmp / 'source.json').read_text()))
        original = next(n for n in source['nodeChanges'] if n.get('name') == 'magnets_learning_compact')
        b = TreeBuilder(source, args.font_dir)
        page = b.byid[gid(original['parentIndex']['guid'])]
        top = [n for n in source['nodeChanges'] if n.get('parentIndex', {}).get('guid') == page['guid']]
        y = max(n['transform']['m12'] + n['size']['y'] for n in top) + 140
        x = min(n['transform']['m02'] for n in top)
        roots, insets, maps = [], [], []
        for i, name in enumerate(VARIANTS):
            root, mapping = clone_frame(b, source, original, page, PREFIX + name, x + i * 1949, y)
            roots.append(root); maps.append(mapping)
            if i == 0:
                inset = growth(b, root, edges)
            elif i == 1:
                inset = lineages(b, root, edges)
            else:
                inset = selection(b, root, edges, candidates, pulls)
            insets.append(inset)
        message = {k: copy.deepcopy(v) for k, v in source.items()
                   if k in ['type', 'sessionID', 'ackID', 'nodeChangeOrder', 'nodeChanges', 'blobs']}
        message['nodeChanges'] += b.nodes
        message['blobs'] = b.blobs
        (tmp / 'combined.json').write_text(json.dumps(message))
        subprocess.run([bun, str(IO), 'encode', str(tmp / 'reference.canvas'), str(tmp / 'combined.json'),
                        str(tmp / 'combined.canvas'), str(args.modules)], check=True)
        saved = json.loads((tmp / 'combined.canvas.json').read_text())
        checks = validate(saved, source, b, roots, insets, maps, edges)
        for name, root, inset in zip(VARIANTS, roots, insets):
            render(saved, gid(root['guid']), OUT / f'magnets_rex_{name}.png', OUT / f'magnets_rex_{name}.pdf')
            render(saved, gid(inset['guid']), OUT / f'rex_{name}_inset.png', OUT / f'rex_{name}_inset.pdf', OUT / f'rex_{name}_inset.svg')
        comparison()
        staged = OUT / 'learning_evolution_fig_with_rex_options.fig'
        package(args.source, tmp / 'combined.canvas', staged, OUT / 'magnets_rex_growth.png', roots)
        report = dict(source=str(args.source.relative_to(ROOT)), source_sha256_before=before,
            output=str(staged.relative_to(ROOT)), checks=checks,
            inset_bounds=dict(x=IX, y=IY, width=IW, height=IH),
            variants=[dict(name=r['name'], root_id=gid(r['guid']), inset_id=gid(i['guid']),
                           position=r['transform']) for r, i in zip(roots, insets)],
            candidate_path=evidence['candidate_path'], process_path=evidence['process_path'],
            edges=[dict(parent=a, child=z, iteration=z) for a, z in edges],
            lineages=evidence['lineages'],
            selection_before_iteration_8={str(i): dict(train_score=candidates[i]['train_score'],
                prior_expansions=pulls[i]) for i in [4, 6]},
            selected_parent_at_iteration_8=process[-1]['selected'],
            new_native_node_ids=[gid(n['guid']) for n in b.nodes],
            extra_inter_319_glyphs=b.extra_glyphs,
            algorithm_reference='https://arxiv.org/pdf/2405.17503',
            implementation='offline_learning/invdyn_core.py:RExPureCandidateSelector')
        if args.apply:
            backup = OUT / 'learning_evolution_fig_before_rex.fig'
            if not backup.exists():
                shutil.copy2(args.source, backup)
            assert hashlib.sha256(args.source.read_bytes()).hexdigest() == before
            pending = args.source.with_name(args.source.name + '.tmp')
            shutil.copy2(staged, pending)
            pending.replace(args.source)
            report['applied_to_source'] = True
            report['source_sha256_after'] = hashlib.sha256(args.source.read_bytes()).hexdigest()
            report['backup'] = str(backup.relative_to(ROOT))
        else:
            report['applied_to_source'] = False
        (OUT / 'rex_tree_search_evidence.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k not in ['new_native_node_ids', 'extra_inter_319_glyphs']}, indent=2))


if __name__ == '__main__':
    main()
