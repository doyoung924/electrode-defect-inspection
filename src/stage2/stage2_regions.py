import sys; sys.path.insert(0,'/home/claude'); sys.path.insert(0,'/home/claude/spec')
from stage2_geo import *
from scipy import ndimage as nd
from common import load, ALL
S=json.load(open('/home/claude/rv2/results_B/summary.json'))
THR={f:S['folds'][f]['strict']['threshold'] for f in S['folds']}
holes={}
for h in csv.DictReader(open(f'{KIT}/holes.csv',encoding='utf-8-sig')): holes.setdefault(h['시편'],[]).append(h)
KC=0.75; CELL=STRIDE/SCALE  # 4 px
Z={f:np.load(f'/mnt/user-data/uploads/scores_B_{f}.npz') for f in THR}
ids=sorted(set().union(*[set(z.files) for z in Z.values()]))
regs=[]
for sid in ids:
    r=rows[sid]
    a=np.asarray(load(r['file'].replace('.HEIC','')).convert('RGB')).astype(np.float32)
    paper=np.array([float(r['paper_r']),float(r['paper_g']),float(r['paper_b'])])
    g=(a/paper).mean(2); sat=(a.max(2)>=250)
    for f,z in Z.items():
        if sid not in z.files: continue
        sc=z[sid].astype(np.float32)/THR[f]; v=valid_cells(sid,sc.shape); sc[~v]=0
        cy,cx=cell_centers(sid,sc.shape)
        dist=nd.distance_transform_edt(v)*CELL/PPM
        lab,n=nd.label(sc>KC,structure=np.ones((3,3)))
        for i,sl in enumerate(nd.find_objects(lab),1):
            m=lab[sl]==i; yy,xx=np.where(m); yy=yy+sl[0].start; xx=xx+sl[1].start
            py=cy[yy,xx]; px=cx[yy,xx]
            area=len(yy)*(CELL/PPM)**2
            P=np.c_[py,px]/PPM; C=np.cov(P.T) if len(yy)>1 else np.eye(2)*0
            w,vec=np.linalg.eigh(C+np.eye(2)*(CELL/PPM)**2/12)
            elong=float(np.sqrt(w[1]/w[0])); vert=float(abs(vec[0,1]))
            # pixel region (each cell covers CELL x CELL px)
            pm=np.zeros(g.shape,bool)
            for yv,xv in zip(py,px):
                pm[int(yv-CELL/2):int(yv+CELL/2)+1,int(xv-CELL/2):int(xv+CELL/2)+1]=True
            Y0,Y1,X0,X1=int(py.min())-40,int(py.max())+41,int(px.min())-40,int(px.max())+41
            sub=g[Y0:Y1,X0:X1]; pmS=pm[Y0:Y1,X0:X1]
            ring=nd.binary_dilation(pmS,iterations=12)&~nd.binary_dilation(pmS,iterations=3)
            bmax=float(sub[pmS].max()); bmean=float(sub[pmS].mean()); bg=float(np.median(sub[ring])) if ring.any() else float(np.median(sub))
            hit=[]
            for h in holes.get(sid,[]):
                rr=(float(h['지름_mm'])/2+1)*PPM
                if np.any((py-float(h['중심y_px']))**2+(px-float(h['중심x_px']))**2<=rr*rr): hit.append(h['구멍'])
            regs.append(dict(fold=f,id=sid,kind=r['kind'],grade=r['grade'],role=r['role'],
                n=len(yy),area=area,eqd=2*np.sqrt(area/np.pi),elong=elong,vert=vert,
                peak=float(sc[yy,xx].max()),mean=float(sc[yy,xx].mean()),edge=float(dist[yy,xx].min()),
                bmax=bmax,bmean=bmean,con=bmax-bg,cmean=bmean-bg,sat=float(sat[Y0:Y1,X0:X1][pmS].mean()),
                cy=float(py.mean()),cx=float(px.mean()),holes=hit))
    print(sid, sum(1 for x in regs if x['id']==sid), flush=True)
pickle.dump(regs,open('/home/claude/regions.pkl','wb'))
