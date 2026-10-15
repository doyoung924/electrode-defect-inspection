import pickle, numpy as np, collections
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
regs=pickle.load(open('regions.pkl','rb')); H,I,S=pickle.load(open('pc_B.pkl','rb'))
FEAT=['eqd','elong','peak','mean','bmax','con','cmean','sat']
FEATX=FEAT+['eqd^2']
def X_of(rs): 
    X=np.array([[r[k] for k in FEAT] for r in rs],float)
    X[:,0]=np.log(X[:,0]); X[:,1]=np.log(X[:,1])
    return np.c_[X, X[:,0]**2]   # size^2: holes have a size range; regions far larger than any hole are not pinholes
train=[r for r in regs if r['kind'] in('정상','핀홀')]
y=np.array([1 if r['holes'] else 0 for r in train]); grp=np.array([r['id'] for r in train]); X=X_of(train)
models={'LR':lambda: make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000,C=1.0)),
        'Tree(d3)':lambda: DecisionTreeClassifier(max_depth=3,class_weight='balanced',min_samples_leaf=10,random_state=0)}
P={}
gkf=GroupKFold(n_splits=6)
for name,mk in models.items():
    p=np.zeros(len(train))
    for tr,te in gkf.split(X,y,grp):
        m=mk().fit(X[tr],y[tr]); p[te]=m.predict_proba(X[te])[:,1]
    P[name]=p
    from sklearn.metrics import roc_auc_score
    print(name,'region AUROC (grouped CV)',round(roc_auc_score(y,p),3))
# defect (이물/줄무늬) regions: model trained on all normal+pinhole regions
other=[r for r in regs if r['kind'] in('이물','줄무늬')]
for name,mk in models.items():
    m=mk().fit(X,y); po=m.predict_proba(X_of(other))[:,1]
    for r,pp in zip(other,po): r.setdefault('p',{})[name]=pp
    if name=='LR':
        coef=m[-1].coef_[0]; print('LR coef', dict(zip(FEATX,np.round(coef,2))))
    else:
        from sklearn.tree import export_text; print(export_text(m,feature_names=FEATX))
for name in P:
    for r,pp in zip(train,P[name]): r.setdefault('p',{})[name]=pp
pickle.dump(train+other,open('regions_p2.pkl','wb'))
