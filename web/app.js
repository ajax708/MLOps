import { predict, toRecord, explain } from './engine.js';
import { buildPlan, planHtml, heatmapHtml, startChallenge } from './extras.js';

const MODELS = [
  { id: 'random_forest', name: 'Random Forest', short: 'Random Forest', color: 'var(--m1)', hex: '#0086A8' },
  { id: 'hist_gb', name: 'Gradient Boosting', short: 'Gradient Boosting', color: 'var(--m2)', hex: '#C2570C' },
  { id: 'logreg', name: 'Regresión logística', short: 'Regresión logística', color: 'var(--m3)', hex: '#6741D9' },
];
const CHOICES = {
  gender: [[2, 'Mujer'], [1, 'Hombre']],
  education: [[1, 'Posgrado'], [2, 'Universidad'], [3, 'Secundaria'], [4, 'Otros']],
  marital: [[1, 'Casado/a'], [2, 'Soltero/a'], [3, 'Otro']],
};
const PAY = [[-1, 'Pagó todo', '#1E7B45'], [0, 'Pagó el mínimo', '#4C9A6A'], [1, '1 mes tarde', '#C98A12'], [2, '2 meses tarde', '#D2691E'], [3, '3+ meses tarde', '#B3261E']];
const MONTHS = ['Último mes', 'Hace 2 meses', 'Hace 3 meses', 'Hace 4 meses', 'Hace 5 meses', 'Hace 6 meses'];
const ORDER = [5, 4, 3, 2, 1, 0]; // de izquierda a derecha: del mes más antiguo al último
const PRESETS = [
  { name: 'Cliente al día', v: { age: 35, limit: 120000, gender: 2, education: 2, marital: 1, months: [0, 0, 0, 0, 0, 0], bill: 40000, paid: 2000 } },
  { name: 'Paga todo siempre', v: { age: 42, limit: 300000, gender: 1, education: 1, marital: 1, months: [-1, -1, -1, -1, -1, -1], bill: 18000, paid: 18000 } },
  { name: 'Un tropiezo', v: { age: 29, limit: 80000, gender: 2, education: 2, marital: 2, months: [2, 0, 0, 0, 0, 0], bill: 50000, paid: 2000 } },
  { name: 'Al límite del cupo', v: { age: 31, limit: 50000, gender: 1, education: 3, marital: 2, months: [0, 0, 0, 0, 0, 0], bill: 48000, paid: 1500 } },
  { name: 'Atrasos seguidos', v: { age: 38, limit: 60000, gender: 1, education: 2, marital: 1, months: [3, 2, 2, 2, 2, 2], bill: 55000, paid: 0 } },
];

const $ = (id) => document.getElementById(id);
const money = (v) => '$ ' + Math.round(v).toLocaleString('es-CO');
const pct = (v, d = 1) => (v * 100).toFixed(d).replace('.', ',') + ' %';
const band = (s) => (s < 30 ? 'Riesgo bajo' : s < 50 ? 'Riesgo medio' : s < 70 ? 'Riesgo alto' : 'Riesgo muy alto');

let state = { ...PRESETS[0].v, months: [...PRESETS[0].v.months] };
let selected = 'random_forest';
let loaded = {};
let info = null;
let plan = null;
const VIEWS = ['simulador', 'modelos', 'datos', 'reto', 'proyecto'];

/* ---------- Controles ---------- */
function buildControls() {
  PRESETS.forEach((p, i) => {
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = p.name;
    b.addEventListener('click', () => { state = { ...p.v, months: [...p.v.months] }; syncControls(); markPreset(i); update(); });
    $('presets').append(b);
  });
  Object.entries(CHOICES).forEach(([key, opts]) => {
    opts.forEach(([v, label]) => {
      const b = document.createElement('button');
      b.type = 'button'; b.textContent = label; b.dataset.v = v;
      b.addEventListener('click', () => { state[key] = v; change(); });
      $(key).append(b);
    });
  });
  const heads = ORDER.map((i) => `<span class="cal-head">${i === 0 ? 'Último<br>mes' : `Hace<br>${i + 1}`}</span>`).join('');
  const rows = PAY.map(([v, label, color]) => `<span class="cal-row" style="--c:${color}">${label}</span>` + ORDER.map((i) =>
    `<button type="button" class="cal-cell" data-m="${i}" data-v="${v}" style="--c:${color}" aria-label="${MONTHS[i]}: ${label}"></button>`).join('')).join('');
  $('calendar').innerHTML = `<span></span>${heads}${rows}<svg class="cal-line" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><polyline id="calPath" vector-effect="non-scaling-stroke"/></svg>`;
  $('calendar').addEventListener('click', (e) => {
    const c = e.target.closest('.cal-cell');
    if (c) { state.months[+c.dataset.m] = +c.dataset.v; change(); }
  });
  ['age', 'limit', 'bill', 'paid'].forEach((k) => $(k).addEventListener('input', (e) => { state[k] = +e.target.value; change(); }));
  $('allGood').addEventListener('click', () => { state.months = [0, 0, 0, 0, 0, 0]; change(); });
  $('allLate').addEventListener('click', () => { state.months = state.months.map(() => 2); change(); });
}
function markPreset(i) { [...$('presets').children].forEach((b, k) => b.classList.toggle('on', k === i)); }
function change() { markPreset(-1); syncControls(); update(); }

function syncControls() {
  ['age', 'limit', 'bill', 'paid'].forEach((k) => { $(k).value = state[k]; });
  $('ageOut').textContent = state.age + ' años';
  $('limitOut').textContent = money(state.limit);
  $('billOut').textContent = money(state.bill);
  $('paidOut').textContent = money(state.paid);
  Object.keys(CHOICES).forEach((k) => [...$(k).children].forEach((b) => b.setAttribute('aria-pressed', +b.dataset.v === state[k])));
  document.querySelectorAll('.cal-cell').forEach((c) => c.setAttribute('aria-pressed', state.months[+c.dataset.m] === +c.dataset.v));
  $('calPath').setAttribute('points', ORDER.map((m, x) => {
    const row = PAY.findIndex(([v]) => v === state.months[m]);
    return `${((x + .5) / 6) * 100},${((row + .5) / PAY.length) * 100}`;
  }).join(' '));
}

/* ---------- Simulador ---------- */
function scores(record) {
  return Object.fromEntries(MODELS.filter((m) => loaded[m.id]).map((m) => [m.id, predict(loaded[m.id], record) * 100]));
}
function update() {
  if (!loaded[selected]) return;
  const record = toRecord(state);
  const s = scores(record);
  const score = s[selected];
  const model = MODELS.find((m) => m.id === selected);

  // Verde azulado -> ámbar -> rojo, según el puntaje
  const hue = score < 50 ? 172 - (score / 50) * 132 : 40 - ((score - 50) / 50) * 50;
  $('card').style.setProperty('--hue', (hue + 360) % 360);
  $('cardScore').textContent = Math.round(score);
  $('cardBand').textContent = band(score);
  $('cardModel').textContent = model.short;
  const use = state.limit ? state.bill / state.limit : 0;
  $('cardUse').textContent = Math.round(use * 100) + ' %';
  $('cardUseBar').style.width = Math.min(use, 1) * 100 + '%';
  $('cardLimit').textContent = '$ ' + (state.limit / 1000).toLocaleString('es-CO') + ' mil';
  $('cardHolder').textContent = `${CHOICES.gender.find((c) => c[0] === state.gender)[1]}, ${state.age} años`;

  $('modelRows').innerHTML = MODELS.map((m) => {
    const v = s[m.id];
    return `<button type="button" class="model-row" data-id="${m.id}" aria-pressed="${m.id === selected}">
      <span class="name"><span class="dot" style="background:${m.color}"></span>${m.short}</span>
      <span class="meter"><span style="width:${v ?? 0}%;background:${m.color}"></span></span>
      <span class="val">${v === undefined ? '…' : Math.round(v)}</span></button>`;
  }).join('');

  const reasons = explain(loaded[selected], record, info.reference).slice(0, 4);
  const max = Math.max(12, ...reasons.map((r) => Math.abs(r.points)));
  $('reasons').innerHTML = reasons.map((r) => {
    const w = (Math.abs(r.points) / max) * 50;
    const sign = r.points >= 0 ? '+' : '−';
    return `<div class="reason"><span>${r.name}</span><span class="track"><span class="bar ${r.points >= 0 ? 'up' : 'down'}" style="width:${w}%"></span></span><span class="val">${sign}${Math.abs(r.points).toFixed(0)}</span></div>`;
  }).join('');

  plan = buildPlan(loaded[selected], state);
  $('plan').innerHTML = planHtml(plan);
  $('maps').innerHTML = MODELS.filter((m) => loaded[m.id]).map((m, i) => heatmapHtml(loaded[m.id], m, state, i === 0)).join('');
  renderOpinions(s);
}
function renderOpinions(s) {
  $('opinions').innerHTML = MODELS.map((m) => `<div class="opinion" style="--c:${m.color}"><div class="name">${m.name}</div><div class="score">${s[m.id] === undefined ? '…' : Math.round(s[m.id])}</div><div class="sub">${s[m.id] === undefined ? 'Cargando' : band(s[m.id])}</div></div>`).join('');
  const v = Object.values(s);
  if (v.length === 3) {
    const gap = Math.round(Math.max(...v) - Math.min(...v));
    $('agreement').textContent = gap <= 5 ? `Los modelos coinciden: solo ${gap} puntos de diferencia.` : gap <= 15 ? `Los modelos difieren en ${gap} puntos.` : `Los modelos discrepan: ${gap} puntos de diferencia.`;
  }
}

/* ---------- Comparar modelos ---------- */
function rocChart() {
  const W = 480, H = 400, L = 62, B = 42, T = 12, R = 26;
  const x = (v) => L + v * (W - L - R), y = (v) => H - B - v * (H - B - T);
  let o = '';
  [0, .25, .5, .75, 1].forEach((v) => {
    o += `<line x1="${x(v)}" x2="${x(v)}" y1="${T}" y2="${H - B}" stroke="#E4EAEE"/><line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="#E4EAEE"/>`;
    o += `<text x="${x(v)}" y="${H - B + 18}" text-anchor="middle">${v * 100}%</text><text x="${L - 7}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`;
  });
  o += `<line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="#8695A0" stroke-dasharray="5 4"/>`;
  [...MODELS].reverse().forEach((m) => {
    o += `<polyline fill="none" stroke="${m.hex}" stroke-width="2.2" points="${info.models[m.id].roc.map(([a, b]) => `${x(a).toFixed(1)},${y(b).toFixed(1)}`).join(' ')}"><title>${m.name}: ROC-AUC ${String(info.models[m.id].auc).replace('.', ',')}</title></polyline>`;
  });
  o += `<text x="${L + (W - L - R) / 2}" y="${H - 4}" text-anchor="middle">Buenos clientes en alerta por error</text>`;
  o += `<text transform="translate(13 ${(H - B + T) / 2}) rotate(-90)" text-anchor="middle">Morosos detectados</text>`;
  const legend = MODELS.map((m) => `<span><i class="dot" style="background:${m.color}"></i>${m.short} · ROC-AUC ${String(info.models[m.id].auc).replace('.', ',')}</span>`).join('');
  $('roc').innerHTML = `<div class="legend">${legend}</div><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Curvas ROC de los tres modelos">${o}</svg>`;
}
function cutRows() {
  const cut = +$('cut').value;
  $('cutOut').textContent = cut + ' o más';
  const pos = info.testPositives, neg = info.testN - pos;
  const best = info.models.random_forest.thresholds.find((r) => Math.round(r[0] * 100) === cut);
  $('cutSummary').innerHTML = `Con Random Forest: de cada 100 clientes que no pagaron, se detectan <b>${Math.round(best[1] / pos * 100)}</b>. De cada 100 que sí pagaron, <b>${Math.round(best[2] / neg * 100)}</b> quedan en alerta por error.`;
  $('cutRows').innerHTML = MODELS.map((m) => {
    const row = info.models[m.id].thresholds.find((r) => Math.round(r[0] * 100) === cut);
    const [, tp, fp] = row;
    return `<div class="cut-row"><div class="name"><span class="dot" style="background:${m.color}"></span>${m.short}</div>
      <div class="pair"><span>Morosos detectados</span><span class="meter"><span style="width:${tp / pos * 100}%;background:${m.color}"></span></span><span class="val">${Math.round(tp / pos * 100)} %</span></div>
      <div class="pair"><span>Buenos clientes en alerta</span><span class="meter"><span style="width:${fp / neg * 100}%;background:#8695A0"></span></span><span class="val">${Math.round(fp / neg * 100)} %</span></div></div>`;
  }).join('');
}
function importanceChart() {
  const groups = Object.keys(info.models.random_forest.importance).sort((a, b) => info.models.random_forest.importance[b] - info.models.random_forest.importance[a]);
  const W = 1000, L = 170, R = 56, rowH = 44, H = rowH * groups.length + 6;
  const max = Math.max(...MODELS.flatMap((m) => Object.values(info.models[m.id].importance)));
  const x = (v) => L + Math.max(v, 0) / max * (W - L - R);
  let o = '';
  groups.forEach((g, i) => {
    const y = i * rowH + 4;
    o += `<text x="${L - 12}" y="${y + 23}" text-anchor="end" class="strong" style="font-size:15px">${g}</text>`;
    MODELS.forEach((m, k) => {
      const v = info.models[m.id].importance[g];
      o += `<rect x="${L}" y="${y + 4 + k * 12}" width="${x(v) - L}" height="9" fill="${m.hex}"><title>${m.short}: ${v.toFixed(3).replace('.', ',')}</title></rect>`;
    });
  });
  const legend = MODELS.map((m) => `<span><i class="dot" style="background:${m.color}"></i>${m.short}</span>`).join('');
  $('importance').innerHTML = `<div class="legend">${legend}</div><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Importancia de cada grupo de datos por modelo">${o}</svg>`;
}

/* ---------- Datos ---------- */
function barChart(title, note, rows, overall) {
  const W = 520, L = 124, R = 54, rowH = 38, T = 22, H = rowH * rows.length + T + 4;
  const max = Math.max(...rows.map((r) => r.rate), overall) * 1.04;
  const x = (v) => L + v / max * (W - L - R);
  let o = `<line x1="${x(overall)}" x2="${x(overall)}" y1="${T - 4}" y2="${H}" stroke="#121B24" stroke-dasharray="4 3"/><text x="${x(overall)}" y="${T - 9}" text-anchor="middle">Promedio ${pct(overall, 0)}</text>`;
  rows.forEach((r, i) => {
    const y = T + i * rowH;
    const hot = r.rate >= overall * 1.4;
    o += `<text x="${L - 10}" y="${y + 20}" text-anchor="end" class="strong">${r.label}</text>`;
    o += `<rect x="${L}" y="${y + 6}" width="${x(r.rate) - L}" height="20" rx="3" fill="${hot ? '#C2570C' : '#0086A8'}"><title>${r.label}: ${pct(r.rate)} de ${r.n.toLocaleString('es-CO')} clientes</title></rect>`;
    o += `<text x="${x(r.rate) + 7}" y="${y + 21}" class="strong">${pct(r.rate, 0)}</text>`;
  });
  return `<div class="block"><h2>${title}</h2><p class="hint">${note}</p><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}">${o}</svg></div>`;
}
function dataTab() {
  const p = info.portfolio;
  $('kpis').innerHTML = [
    [p.n.toLocaleString('es-CO'), 'clientes de tarjeta'],
    [pct(p.rate), 'no pagó el mes siguiente'],
    [pct(p.pay[0].rate, 0), 'de impago entre quienes están al día'],
    [pct(p.pay[2].rate, 0), 'de impago con 2 meses de atraso'],
  ].map(([b, s]) => `<div class="kpi"><b>${b}</b><span>${s}</span></div>`).join('');
  const late = p.lateMonths.map((r) => ({ ...r, label: r.label === '0' ? 'Ninguno' : r.label === '1' ? '1 mes' : `${r.label} meses` }));
  $('charts').innerHTML = [
    barChart('El atraso del último mes lo dice casi todo', 'Porcentaje de clientes que no pagó el mes siguiente.', p.pay, p.rate),
    barChart('Cada mes de atraso suma riesgo', 'Impago según cuántos de los últimos 6 meses tuvo atraso.', late, p.rate),
    barChart('A mayor cupo, menos impago', 'Impago según el cupo de la tarjeta, en dólares.', p.limit, p.rate),
    barChart('La edad cambia poco', 'Impago según el grupo de edad.', p.age, p.rate),
  ].join('');
}

/* ---------- Tarjeta y navegación ---------- */
function tilt() {
  if (matchMedia('(prefers-reduced-motion:reduce)').matches || matchMedia('(hover:none)').matches) return;
  const stage = $('stage'), card = $('card');
  stage.addEventListener('pointermove', (e) => {
    const r = stage.getBoundingClientRect();
    const dx = (e.clientX - r.left) / r.width - .5, dy = (e.clientY - r.top) / r.height - .5;
    card.style.transform = `rotateY(${dx * 9}deg) rotateX(${-dy * 9}deg)`;
  });
  stage.addEventListener('pointerleave', () => { card.style.transform = ''; });
}
function go(view) {
  VIEWS.forEach((v) => {
    $(v).hidden = v !== view;
    $(`tab-${v}`).setAttribute('aria-selected', v === view);
  });
  window.scrollTo(0, 0);
}
function route() { const h = location.hash.slice(1); go(VIEWS.includes(h) ? h : 'simulador'); }

async function loadJson(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(path);
  return r.json();
}
async function start() {
  buildControls(); syncControls(); markPreset(0); tilt(); route();
  addEventListener('hashchange', route);
  document.querySelectorAll('[role=tab]').forEach((t) => t.addEventListener('click', () => { location.hash = t.id.replace('tab-', ''); }));
  $('modelRows').addEventListener('click', (e) => {
    const b = e.target.closest('.model-row');
    if (b && loaded[b.dataset.id]) { selected = b.dataset.id; update(); }
  });
  $('cut').addEventListener('input', cutRows);
  $('plan').addEventListener('click', (e) => {
    if (e.target.id === 'applyPlan' && plan) { state = { ...plan.target, months: [...plan.target.months] }; change(); }
  });
  renderOpinions({});
  try {
    [info, loaded.hist_gb, loaded.logreg] = await Promise.all(['data/insights.json', 'models/hist_gb.json', 'models/logreg.json'].map(loadJson));
    rocChart(); cutRows(); importanceChart(); dataTab();
    selected = 'hist_gb'; update();
    // Random Forest es el archivo pesado: llega después, sin bloquear la página
    loaded.random_forest = await loadJson('models/random_forest.json');
    selected = 'random_forest'; update();
    startChallenge($('challenge'), await loadJson('data/challenge.json'), loaded.random_forest);
    document.body.dataset.ready = '1';
  } catch (err) {
    $('cardBand').textContent = 'Sin datos';
    $('reasons').innerHTML = '<p class="hint">No se pudieron cargar los modelos. Recarga la página.</p>';
    console.error(err);
  }
}
start();
