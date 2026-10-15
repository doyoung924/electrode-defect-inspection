import pickle, numpy as np, collections, json
regs=pickle.load(open('regions_p2.pkl','rb')); H,I,S=pickle.load(open('pc_B.pkl','rb'))
FOLDS=sorted(S['folds'])
tests={f:[x for x in I if x['fold']==f] for f in FOLDS}   # images tested per fold
grp=lambda x:'A' if x['role'].startswith('A_fold') else ('GL' if x['role']=='A_excluded' else x['grade'] if x['kind']=='정상' else x['kind'])
R=collections.defaultdict(list)
for r in regs: R[(r['fold'],r['id'])].append(r)
evh=[h for h in H if h['flag']=='평가']
def evaluate(alarm):   # alarm(region)->bool
    out=collections.Counter(); n=collections.Counter()
    for f in FOLDS:
        for x in tests[f]:
            g=grp(x); a=any(alarm(r) for r in R[(f,x['id'])])
            n[g]+=1; out[g]+=a
    det={}; 
    for h in evh:
        hit=any(alarm(r) for r in R[(h['fold'],h['id'])] if h['hole'] in r['holes'])
        b='1-1.5' if h['d']<1.5 else '1.5-2' if h['d']<2 else '2-2.5' if h['d']<2.5 else '2.5-3' if h['d']<3 else '3+'
        det.setdefault(b,[]).append(hit); det.setdefault('all',[]).append(hit)
    normal_fp=(out['A']+out['B']+out['GL'])/(n['A']+n['B']+n['GL'])
    return dict(fp_all=normal_fp,fpA=(out['A'],n['A']),fpB=(out['B'],n['B']),fpGL=(out['GL'],n['GL']),
                det={k:float(np.mean(v)) for k,v in det.items()},
                defect={g:(out[g],n[g]) for g in ('이물','줄무늬','핀홀')})
res={'stage1':{},'LR':{}}
for k in np.round(np.arange(0.75,1.61,0.01),2): res['stage1'][float(k)]=evaluate(lambda r,k=k:r['peak']>k)
for t in np.round(np.arange(0.02,0.99,0.01),2): res['LR'][float(t)]=evaluate(lambda r,t=t:r['p']['LR']>t)
pickle.dump(res,open('stage2_res2.pkl','wb'))
def show(tag,e): print(f"{tag}: normalFP {e['fp_all']:.2f} (A {e['fpA'][0]}/{e['fpA'][1]}, B {e['fpB'][0]}/{e['fpB'][1]}, 광택 {e['fpGL'][0]}/{e['fpGL'][1]}) | holes {e['det']['all']:.2f} | "+' '.join(f"{b}:{e['det'][b]:.2f}" for b in ['1-1.5','1.5-2','2-2.5','2.5-3','3+'])+f" | 이물 {e['defect']['이물']} 줄무늬 {e['defect']['줄무늬']}")
show('stage1 k=1.00 (strict)',res['stage1'][1.0])
show('LR τ=0.50',res['LR'][0.5])
for target in (0.05,0.10,0.15,0.20,0.30):
    for name in ('stage1','LR'):
        best=min(res[name].items(), key=lambda kv:(abs(kv[1]['fp_all']-target), -kv[1]['det']['all']))
        show(f"FP≈{target:.2f} {name}({best[0]})",best[1])
