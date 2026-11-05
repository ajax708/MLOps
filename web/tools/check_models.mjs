import {readFileSync as R} from 'fs';
const J=f=>JSON.parse(R(new URL(f,import.meta.url)));
export function features(p,r){const x=[];p.num.forEach((c,i)=>x.push((r[c]-p.mean[i])/p.scale[i]));p.cat.forEach((c,i)=>p.categories[i].forEach(v=>x.push(r[c]===v?1:0)));return x;}
const sig=z=>1/(1+Math.exp(-z));
export function predict(m,r){const x=features(m.pre,r);
 if(m.kind==='logreg')return sig(m.coef.reduce((s,c,i)=>s+c*x[i],m.intercept));
 if(m.kind==='random_forest'){let s=0;for(const t of m.trees){let i=0;while(Array.isArray(t[i]))i=Math.fround(x[t[i][0]])<=t[i][1]?i+1:t[i][2];s+=t[i];}return s/m.trees.length;}
 let s=m.baseline;for(const t of m.trees){let i=0;while(Array.isArray(t[i]))i=x[t[i][0]]<=t[i][1]?t[i][2]:t[i][3];s+=t[i];}return sig(s);}
const rows=J('./test.json'),exp=J('./expected.json');
function auc(y,p){const a=p.map((v,i)=>[v,y[i]]).sort((u,v)=>u[0]-v[0]);let rank=0,sum=0,i=0,n1=0;while(i<a.length){let j=i;while(j<a.length&&a[j][0]===a[i][0])j++;const rk=(i+1+j)/2;for(let k=i;k<j;k++)if(a[k][1]){sum+=rk;n1++;}i=j;}const n0=a.length-n1;return (sum-n1*(n1+1)/2)/(n1*n0);}
for(const n of ['logreg','hist_gb','random_forest']){const m=J(`./${n}.json`);const t0=performance.now();const p=rows.map(r=>predict(m,r));const ms=(performance.now()-t0)/rows.length;
 const d=p.map((v,i)=>Math.abs(v-exp[n][i]));console.log(n,'dif max',Math.max(...d).toExponential(2),'media',(d.reduce((a,b)=>a+b)/d.length).toExponential(2),'AUC js',auc(rows.map(r=>r.default),p).toFixed(4),'AUC sk',auc(rows.map(r=>r.default),exp[n]).toFixed(4),'ms/cliente',ms.toFixed(3));}
{const m=J('./random_forest.json');const d=rows.map((r,i)=>[Math.abs(predict(m,r)-exp.random_forest[i]),i]).sort((a,b)=>b[0]-a[0]);console.log('filas >1e-6:',d.filter(x=>x[0]>1e-6).length,'de',rows.length,'| >1e-3:',d.filter(x=>x[0]>1e-3).length,'| top',d.slice(0,3).map(x=>x[0].toExponential(1)).join(' '));}
