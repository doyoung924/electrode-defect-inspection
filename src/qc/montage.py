from measure import *
from PIL import ImageDraw
import pickle, sys
frac=0.2
tiles=[]; allres={}
for sid in SEEDS:
    try:
        res,info=measure(sid,frac=frac)
    except Exception as e:
        print(sid,'ERR',e); continue
    a=info['a']; wg=info['wg']
    allres[sid]={k:{kk:vv for kk,vv in v.items() if kk not in('comp',)} for k,v in res.items()}
    for key in ('T','M','LL','LR'):
        v=res[key]; half=int(6*PPM)
        cy,cx=v['y_px'],v['x_px']
        y0,x0=max(0,cy-half),max(0,cx-half)
        crop=a[y0:cy+half,x0:cx+half].mean(2)/wg
        img=np.clip(crop*255*2.0,0,255).astype(np.uint8)
        rgb=np.stack([img]*3,-1)
        comp=v['comp']; ty,tx=v['sy'],v['sx']
        cc=np.zeros(crop.shape,bool)
        oy,ox=ty-y0,tx-x0
        h,w=comp.shape
        ys0,xs0=max(0,oy),max(0,ox); ys1,xs1=min(crop.shape[0],oy+h),min(crop.shape[1],ox+w)
        cc[ys0:ys1,xs0:xs1]=comp[ys0-oy:ys1-oy, xs0-ox:xs1-ox]
        e=cc^nd.binary_erosion(cc,iterations=2)
        rgb[e]=[255,0,0] if v['visible'] else [255,200,0]
        im=Image.fromarray(rgb).resize((180,180))
        d=ImageDraw.Draw(im); d.rectangle([0,0,180,16],fill='black')
        d.text((3,2),f"{sid}-{key} {v['d']}mm{'' if v['visible'] else ' ?'}",fill='white')
        # 1mm scale bar
        bar=int(180/(12)) ; d.line([(5,170),(5+bar*2,170)],fill='cyan',width=3)
        tiles.append(im)
cols=8; rows=(len(tiles)+cols-1)//cols
S_=Image.new('RGB',(cols*182,rows*182),'white')
for k,t in enumerate(tiles): S_.paste(t,((k%cols)*182,(k//cols)*182))
S_.save('holes_montage.jpg',quality=88)
pickle.dump(allres,open('holes_p.pkl','wb'))
for sid,r in allres.items(): print(sid,' | '.join(f"{k}:{v['d']}{'' if v['visible'] else '?'}(snr{round(v['snr'])})" for k,v in r.items()))
