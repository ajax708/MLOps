import { predict, toRecord } from './engine.js';

const money = (v) => '$ ' + Math.round(v).toLocaleString('es-CO');
export const riskColor = (score, light = 42) => {
  const hue = score < 50 ? 172 - (score / 50) * 132 : 40 - ((score - 50) / 50) * 50;
  return `hsl(${(hue + 360) % 360} 66% ${light}%)`;
};

/* ---------- Plan para bajar el riesgo: solo cambios que dependen del cliente ---------- */
const ACTIONS = [
  { label: 'Ponerse al día en todos los meses', can: (s) => s.months.filter((m) => m > 0).length > 1, apply: (s) => ({ ...s, months: s.months.map((m) => Math.min(m, 0)) }) },
  { label: 'Ponerse al día el último mes', can: (s) => s.months[0] > 0, apply: (s) => ({ ...s, months: [0, ...s.months.slice(1)] }) },
  { label: 'Pagar la factura completa', can: (s) => s.paid < s.bill, apply: (s) => ({ ...s, paid: Math.min(s.bill, 100000) }) },
  { label: 'Pagar el doble cada mes', can: (s) => s.paid < 50000, apply: (s) => ({ ...s, paid: Math.max(s.paid * 2, 1000) }) },
  { label: 'Reducir el gasto a la mitad', can: (s) => s.bill >= 2000, apply: (s) => ({ ...s, bill: Math.round(s.bill / 2000) * 1000 }) },
];

export function buildPlan(model, state) {
  const score = (s) => predict(model, toRecord(s)) * 100;
  const start = score(state);
  const steps = [];
  let current = state, now = start;
  for (let i = 0; i < 3 && now >= 30; i++) {
    const best = ACTIONS.filter((a) => a.can(current))
      .map((a) => { const next = a.apply(current); return { label: a.label, next, value: score(next) }; })
      .sort((a, b) => a.value - b.value)[0];
    if (!best || now - best.value < 2) break;
    steps.push({ label: best.label, from: now, to: best.value });
    current = best.next; now = best.value;
  }
  return { start, end: now, steps, target: current };
}

export function planHtml(plan) {
  if (plan.start < 30) return '<p class="plan-ok">Este cliente ya tiene riesgo bajo. No necesita cambios.</p>';
  if (!plan.steps.length) return '<p class="hint">Con los cambios disponibles, el modelo no ve una mejora clara.</p>';
  const rows = plan.steps.map((s, i) => `<li><span class="step-n">${i + 1}</span><span>${s.label}</span><span class="val">${Math.round(s.from)} → <b>${Math.round(s.to)}</b></span></li>`).join('');
  return `<ol class="plan">${rows}</ol><div class="plan-foot"><span>Riesgo final estimado: <b>${Math.round(plan.end)}</b></span><button type="button" class="apply" id="applyPlan">Aplicar plan</button></div>`;
}

/* ---------- Mapa de riesgo: cupo contra atraso del último mes ---------- */
const ROWS = [[3, '3+ meses'], [2, '2 meses'], [1, '1 mes'], [0, 'Mínimo'], [-1, 'Pagó todo']];
const LIMITS = Array.from({ length: 20 }, (_, i) => 20000 + i * 40000);

export function heatmapHtml(model, meta, state, withLabels) {
  const W = withLabels ? 386 : 310, L = withLabels ? 76 : 0, T = 6, cw = 15.4, ch = 37, H = T + ch * ROWS.length + 26;
  let o = '';
  ROWS.forEach(([pay, label], r) => {
    const y = T + r * ch;
    if (withLabels) o += `<text x="${L - 7}" y="${y + ch / 2 + 4}" text-anchor="end">${label}</text>`;
    LIMITS.forEach((limit, c) => {
      const s = predict(model, toRecord({ ...state, limit, months: [pay, ...state.months.slice(1)] })) * 100;
      o += `<rect x="${L + c * cw}" y="${y}" width="${cw - 1}" height="${ch - 1}" fill="${riskColor(s)}"><title>Cupo ${money(limit)}, último mes ${label}: riesgo ${Math.round(s)}</title></rect>`;
    });
  });
  const col = Math.round((Math.min(state.limit, LIMITS[LIMITS.length - 1]) - LIMITS[0]) / 40000);
  const row = ROWS.findIndex(([p]) => p === Math.max(-1, Math.min(3, state.months[0])));
  if (col >= 0 && row >= 0) {
    const cx = L + Math.max(col, 0) * cw + (cw - 1) / 2, cy = T + row * ch + (ch - 1) / 2;
    o += `<circle cx="${cx}" cy="${cy}" r="8" fill="#fff" stroke="#121B24" stroke-width="3"><title>Este cliente</title></circle>`;
  }
  o += `<text x="${L}" y="${H - 8}">$ 20 mil</text><text x="${L + cw * 20 - 1}" y="${H - 8}" text-anchor="end">$ 780 mil</text><text x="${L + cw * 10}" y="${H - 8}" text-anchor="middle">Cupo</text>`;
  return `<div class="map"><h3><span class="dot" style="background:${meta.color}"></span>${meta.name}</h3><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Mapa de riesgo de ${meta.name}">${o}</svg></div>`;
}

/* ---------- Reto: ¿le ganas al modelo? ---------- */
const NAMES = {
  gender: { 1: 'Hombre', 2: 'Mujer' },
  education: { 1: 'con posgrado', 2: 'universitario/a', 3: 'con secundaria', 4: 'otros estudios' },
  marital: { 1: 'casado/a', 2: 'soltero/a', 3: 'otro estado civil' },
};
const shuffle = (a) => a.map((v) => [Math.random(), v]).sort((x, y) => x[0] - y[0]).map((x) => x[1]);
const payCell = (v) => {
  const [txt, bg, fg] = v <= 0 ? ['Al día', '#DDF1E7', '#14532D'] : v === 1 ? ['1 mes', '#FBEBC8', '#7A4A00'] : v === 2 ? ['2 meses', '#F9D5B5', '#7C2D12'] : ['3+ meses', '#F5C2BF', '#7F1D1D'];
  return `<span style="background:${bg};color:${fg}">${txt}</span>`;
};

export function startChallenge(root, clients, model) {
  const ROUNDS = 10;
  let game;
  function newGame() {
    const paid = shuffle(clients.filter((c) => c.default === 0)).slice(0, 5);
    const late = shuffle(clients.filter((c) => c.default === 1)).slice(0, 5);
    game = { list: shuffle([...paid, ...late]), i: 0, you: 0, ai: 0 };
    round();
  }
  function board() {
    return `<div class="board"><span>Cliente <b>${Math.min(game.i + 1, ROUNDS)}</b> de ${ROUNDS}</span><span>Tú <b>${game.you}</b></span><span>Modelo <b>${game.ai}</b></span></div>`;
  }
  function profile(c) {
    const months = [6, 5, 4, 3, 2, 1].map((i) => payCell(c[`PAY_${i}`])).join('');
    return `<div class="ficha">
      <h2>${NAMES.gender[c.SEX]}, ${c.AGE} años, ${NAMES.education[c.EDUCATION]}, ${NAMES.marital[c.MARRIAGE]}</h2>
      <dl><div><dt>Cupo</dt><dd>${money(c.LIMIT_BAL)}</dd></div><div><dt>Última factura</dt><dd>${money(c.BILL_AMT1)}</dd></div><div><dt>Último pago</dt><dd>${money(c.PAY_AMT1)}</dd></div></dl>
      <p class="hint">Sus pagos, del mes más antiguo al último:</p><div class="strip">${months}</div></div>`;
  }
  function round() {
    const c = game.list[game.i];
    root.innerHTML = `${board()}<div class="duel">${profile(c)}<div class="ask"><p>¿Pagó su tarjeta el mes siguiente?</p><button type="button" data-a="0" class="choice yes">Sí pagó</button><button type="button" data-a="1" class="choice no">No pagó</button></div></div>`;
  }
  function reveal(answer) {
    const c = game.list[game.i];
    const score = predict(model, c) * 100;
    const ai = score >= 50 ? 1 : 0;
    const youOk = answer === c.default, aiOk = ai === c.default;
    game.you += youOk ? 1 : 0; game.ai += aiOk ? 1 : 0;
    const said = (v) => (v ? 'No pagó' : 'Sí pagó');
    const mark = (ok) => `<span class="${ok ? 'ok' : 'no'}">${ok ? 'Acertó' : 'Falló'}</span>`;
    root.innerHTML = `${board()}<div class="duel">${profile(c)}<div class="ask"><p class="truth">En la realidad: <b>${said(c.default)}</b></p>
      <table><tr><td>Tú</td><td>${said(answer)}</td><td>${mark(youOk)}</td></tr><tr><td>Modelo <small>riesgo ${Math.round(score)}</small></td><td>${said(ai)}</td><td>${mark(aiOk)}</td></tr></table>
      <button type="button" class="apply" id="nextRound">${game.i + 1 < ROUNDS ? 'Siguiente cliente' : 'Ver resultado'}</button></div></div>`;
  }
  function finish() {
    const msg = game.you > game.ai ? '¡Le ganaste al modelo!' : game.you < game.ai ? 'El modelo te ganó esta vez.' : 'Empate con el modelo.';
    root.innerHTML = `<div class="final"><h2>${msg}</h2><div class="scores"><div><b>${game.you}</b><span>Tú</span></div><div><b>${game.ai}</b><span>Random Forest</span></div></div><p class="hint">Sobre ${ROUNDS} clientes reales que el modelo nunca vio. El modelo dice «no pagó» cuando su riesgo es 50 o más.</p><button type="button" class="apply" id="again">Jugar otra vez</button></div>`;
  }
  root.addEventListener('click', (e) => {
    const choice = e.target.closest('.choice');
    if (choice) return reveal(+choice.dataset.a);
    if (e.target.id === 'nextRound') { game.i += 1; return game.i < ROUNDS ? round() : finish(); }
    if (e.target.id === 'again') newGame();
    return undefined;
  });
  newGame();
}
