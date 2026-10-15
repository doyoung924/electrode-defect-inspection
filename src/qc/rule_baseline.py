from seg import *
import csv, pickle
KIT='kit'
rows=list(csv.DictReader(open(f'{KIT}/specimens.csv',encoding='utf-8')))
holes={}
for h in csv.DictReader(open(f'{KIT}/holes.csv',encoding='utf-8-sig')): holes.setdefault(h['시편'],[]).append(h)
def mask(sid,k): return np.asarray(Image.open(f'{KIT}/masks/{sid}_{k}.png'))>127
OPEN=81; C=0.10
out={}
for r in rows:
    sid=r['id']
    a=np.asarray(load(r['file'].replace('.HEIC','')).convert('RGB')).astype(np.float32)
    paper=np.array([float(r['paper_r']),float(r['paper_g']),float(r['paper_b'])])
    g=(a/paper).mean(2)
    roi=np.kron(mask(sid,'roi'),np.ones((S,S),bool))
    ys,xs=np.where(roi); y0,y1,x0,x1=ys.min()-60,ys.max()+60,xs.min()-60,xs.max()+60
    c=g[y0:y1,x0:x1]; R=roi[y0:y1,x0:x1]
    th=c-nd.grey_opening(c,size=(OPEN,OPEN))
    th[~R]=0
    lab,n=nd.label(th>C); 
    yy,xx=np.mgrid[0:c.shape[0],0:c.shape[1]]
    def blobmax(region):
        ids=np.unique(lab[region&(lab>0)])
        if len(ids)==0: return 0.0
        sz=nd.sum(np.ones_like(lab),lab,ids)
        return float(2*np.sqrt(sz.max()/np.pi)/PPM)
    rec=dict(id=sid,kind=r['kind'],grade=r['grade'],role=r['role'],con=float(th.max()),blob=blobmax(R))
    if sid in holes:
        keep=R.copy(); hs=[]
        for h in holes[sid]:
            cy=float(h['중심y_px'])-y0; cx=float(h['중심x_px'])-x0; d=float(h['지름_mm'])
            m=(yy-cy)**2+(xx-cx)**2 <= ((d/2+1)*PPM)**2
            hs.append(dict(hole=h['구멍'],d=d,flag=h['판정'],con=float(th[m].max()),blob=blobmax(m)))
            keep&=(yy-cy)**2+(xx-cx)**2 > ((d/2+5)*PPM)**2
        rec['holes']=hs; rec['bg_con']=float(th[keep].max()); rec['bg_blob']=blobmax(keep)
    out[sid]=rec
    print(sid,r['kind'],r['grade'],round(rec['con'],3),round(rec['blob'],2),flush=True)
pickle.dump(out,open('/home/claude/rule_scores.pkl','wb'))
