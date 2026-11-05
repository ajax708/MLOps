// Motor de predicción: reproduce los modelos de scikit-learn en el navegador.
// Verificado contra los 6.000 clientes de test con web/tools/check_models.mjs.

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

function features(pre, record) {
  const x = [];
  pre.num.forEach((col, i) => x.push((record[col] - pre.mean[i]) / pre.scale[i]));
  pre.cat.forEach((col, i) => pre.categories[i].forEach((v) => x.push(record[col] === v ? 1 : 0)));
  return x;
}

export function predict(model, record) {
  const x = features(model.pre, record);
  if (model.kind === 'logreg') {
    return sigmoid(model.coef.reduce((sum, c, i) => sum + c * x[i], model.intercept));
  }
  if (model.kind === 'random_forest') {
    let sum = 0;
    for (const tree of model.trees) {
      let i = 0;
      while (Array.isArray(tree[i])) i = Math.fround(x[tree[i][0]]) <= tree[i][1] ? i + 1 : tree[i][2];
      sum += tree[i];
    }
    return sum / model.trees.length;
  }
  let sum = model.baseline;
  for (const tree of model.trees) {
    let i = 0;
    while (Array.isArray(tree[i])) i = x[tree[i][0]] <= tree[i][1] ? tree[i][2] : tree[i][3];
    sum += tree[i];
  }
  return sigmoid(sum);
}

// Traduce los controles de la interfaz a las columnas originales del dataset
export function toRecord(c) {
  const r = { LIMIT_BAL: c.limit, AGE: c.age, SEX: c.gender, EDUCATION: c.education, MARRIAGE: c.marital };
  for (let i = 1; i <= 6; i++) {
    r[`PAY_${i}`] = c.months[i - 1];
    r[`BILL_AMT${i}`] = c.bill;
    r[`PAY_AMT${i}`] = c.paid;
  }
  return r;
}

const GROUPS = {
  'Historial de pagos': [1, 2, 3, 4, 5, 6].map((i) => `PAY_${i}`),
  Cupo: ['LIMIT_BAL'],
  Facturas: [1, 2, 3, 4, 5, 6].map((i) => `BILL_AMT${i}`),
  'Pagos realizados': [1, 2, 3, 4, 5, 6].map((i) => `PAY_AMT${i}`),
  Edad: ['AGE'],
  'Datos personales': ['SEX', 'EDUCATION', 'MARRIAGE'],
};

// ¿Cuánto cambia el puntaje si este grupo de datos fuera el de un cliente típico?
export function explain(model, record, reference) {
  const base = predict(model, record);
  return Object.entries(GROUPS)
    .map(([name, cols]) => {
      const alt = { ...record };
      cols.forEach((col) => { alt[col] = reference[col]; });
      return { name, points: (base - predict(model, alt)) * 100 };
    })
    .sort((a, b) => Math.abs(b.points) - Math.abs(a.points));
}
