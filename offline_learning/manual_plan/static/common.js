// Shared helpers: API calls and grid rendering.
// Autumn renders cells as CSS colour NAMES ("black", "lightblue", "darkgreen", ...), so a
// colour is used verbatim; the map only overrides the few that read badly on a dark page.
const COLOR = {black:'#0d1017', white:'#eef1f5', gray:'#8b929c', grey:'#8b929c'};
const col = c => COLOR[c] || c || '#0d1017';

async function api(path, body, method) {
  const opt = {method: method || (body ? 'POST' : 'GET'),
               headers: {'Content-Type': 'application/json'}};
  if (body) opt.body = JSON.stringify(body);
  const r = await fetch(path, opt);
  if (!r.ok) throw new Error((await r.text()).slice(0, 300));
  return r.json();
}

// gridJson: the renderer's JSON string. opts: {cell, onClick, diff:[[r,c],...]}
function gridEl(gridJson, opts) {
  opts = opts || {};
  const el = document.createElement('div');
  el.className = 'grid' + (opts.onClick ? ' click' : '');
  if (!gridJson) { el.textContent = '(no frame)'; return el; }
  const g = typeof gridJson === 'string' ? JSON.parse(gridJson) : gridJson;
  const n = g[0].length;
  el.style.gridTemplateColumns = `repeat(${n}, ${opts.cell || 22}px)`;
  el.style.setProperty('--cs', (opts.cell || 22) + 'px');
  const dd = new Set((opts.diff || []).map(p => p[0] + ',' + p[1]));
  g.forEach((row, r) => row.forEach((v, c) => {
    const d = document.createElement('div');
    d.style.background = col(v);
    if (dd.has(r + ',' + c)) d.className = 'diff';
    d.title = `${r},${c} ${v}`;
    if (opts.onClick) d.onclick = () => opts.onClick(r, c);
    el.appendChild(d);
  }));
  return el;
}

// Thumbnail for the filmstrip: one canvas per frame instead of size^2 divs.
function canvasEl(gridJson, px) {
  const g = typeof gridJson === 'string' ? JSON.parse(gridJson) : gridJson;
  const n = g[0].length, m = g.length;
  const cv = document.createElement('canvas');
  cv.width = n * px; cv.height = m * px;
  cv.style.imageRendering = 'pixelated';
  const x = cv.getContext('2d');
  g.forEach((row, r) => row.forEach((v, c) => {
    x.fillStyle = col(v); x.fillRect(c * px, r * px, px, px);
  }));
  return cv;
}

const isClick = a => typeof a === 'string' && a.startsWith('click');
const clickRC = a => a.split(/\s+/).slice(1).map(Number);
function el(tag, cls, txt) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt !== undefined) e.textContent = txt;
  return e;
}
