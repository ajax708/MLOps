import sys,json,gzip,warnings,joblib,numpy as np,pandas as pd
warnings.filterwarnings("ignore")
sp=sys.argv[1]; out=f"{sp}/v2/js"
d=pd.read_csv("data/reference/test.csv")
r6=lambda v: float(v); r5=lambda v: float(v)
def pre(m):
    p=m.named_steps["preprocess"]; num=p.named_transformers_["num"]; cat=p.named_transformers_["cat"]
    cols={t:list(c) for t,_,c in p.transformers_ if t!="remainder"}
    # num puede ser pipeline o scaler
    sc=num if hasattr(num,"mean_") else num[-1]
    return {"input":list(m.feature_names_in_),"num":cols["num"],"mean":[float(x) for x in sc.mean_],"scale":[float(x) for x in sc.scale_],
            "cat":cols["cat"],"categories":[[int(c) for c in cs] for cs in cat.categories_],"drop":None if cat.drop is None else str(cat.drop)}
res={}
for n in ["logreg","hist_gb","random_forest"]:
    m=joblib.load(f"{sp}/v2/models/{n}.joblib"); c=m.named_steps["model"]; o={"kind":n,"pre":pre(m)}
    if n=="logreg":
        o["coef"]=[float(x) for x in c.coef_[0]]; o["intercept"]=float(c.intercept_[0])
    elif n=="random_forest":
        trees=[]
        for e in c.estimators_:
            t=e.tree_; v=t.value[:,0,:]; v=v/v.sum(1,keepdims=True)
            # recorrido en profundidad: hijo izquierdo = i+1; se guarda solo el derecho
            trees.append([[int(t.feature[i]),r6(t.threshold[i]),int(t.children_right[i])] if t.children_left[i]!=-1 else r5(v[i,1]) for i in range(t.node_count)])
        o["trees"]=trees
    else:
        o["baseline"]=float(np.ravel(c._baseline_prediction)[0]); trees=[]
        for g in c._predictors:
            nd=g[0].nodes
            trees.append([[int(x["feature_idx"]),float(x["num_threshold"]),int(x["left"]),int(x["right"])] if not x["is_leaf"] else float(x["value"]) for x in nd])
        o["trees"]=trees
    s=json.dumps(o,separators=(",",":")); open(f"{out}/{n}.json","w").write(s)
    q=m.predict_proba(d[list(m.feature_names_in_)])[:,1]; res[n]=[float(x) for x in q]
    print(f"{n}: {len(s)/1e6:.3f} MB, gzip {len(gzip.compress(s.encode(),6))/1e6:.3f} MB")
    if n=="hist_gb": print(" faltantes/categ:", any(x['is_categorical'] for g in c._predictors for x in g[0].nodes), "iters",len(c._predictors))
d.to_json(f"{out}/test.json",orient="records"); json.dump(res,open(f"{out}/expected.json","w"))
