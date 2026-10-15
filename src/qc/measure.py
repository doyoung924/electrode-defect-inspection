from holes import *
SEEDS={
'P01':{'T':(21,32),'M':(50.7,33.2),'LL':(101.5,21.5),'LR':(102.1,50.2)},
'P02':{'T':(18,26),'M':(43.8,24.5),'LL':(78.7,9.8),'LR':(78.1,38.4)},
'P03':{'T':(18.7,22.7),'M':(46.2,25.4),'LL':(88.4,10.4),'LR':(89.6,42.3)},
'P04':{'T':(16.9,30.7),'M':(46.7,29.5),'LL':(84.6,13.9),'LR':(85.4,39.8)},
'P05':{'T':(19.2,19.1),'M':(49.0,20.7),'LL':(89.6,6.0),'LR':(92.7,33.8)},
'P06':{'T':(18.8,20.5),'M':(46.9,23.2),'LL':(88.4,12.5),'LR':(86.2,38.2)},
'P07':{'T':(20.2,19.9),'M':(45.0,21.7),'LL':(102.2,16.1),'LR':(101,39)},
'P08':{'T':(20.7,38.9),'M':(51.5,34.0),'LL':(98.8,22.3),'LR':(101.6,43.8)},
'P09':{'T':(22.7,24.1),'M':(50.2,25.2),'LL':(105.8,14.8),'LR':(107.2,40.0)},
'P11':{'T':(19.8,24.9),'M':(48.8,28.5),'LL':(97.0,17.5),'LR':(100.4,48.2)},
'P12':{'T':(19.5,21.5),'M':(41.4,24.0),'LL':(88.5,16.3),'LR':(89,40)},
}
LOWCUT={('P12','T'):2.5,('P12','M'):8}
def measure(sid, win_mm=7, open_px=81, frac=0.2):
    a,sm=gray_small(sid); bm,_=band_mask(sm)
    pmed,sat=paper_med(sm); wg=float(pmed.mean())
    B=np.kron(bm,np.ones((S,S),bool)); Bi=nd.binary_erosion(B,iterations=20)
    ys,xs=np.where(B); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    g=a.mean(2)/wg; c=g[y0:y1,x0:x1]
    th=c-nd.grey_opening(c,size=(open_px,open_px))
    _,(thr,thr_hi,band_lv,foil_lv)=band_mask(sm)
    band_n=band_lv/wg; foil_n=foil_lv/wg
    v=th[Bi[y0:y1,x0:x1]]; med=np.median(v); sig=np.median(np.abs(v-med))*1.4826
    res={}
    Bc=Bi[y0:y1,x0:x1]
    for key,(ym,xm) in SEEDS[sid].items():
        cy,cx=int(ym*PPM),int(xm*PPM); w=int(win_mm*PPM)
        sy,sx=max(0,cy-w),max(0,cx-w)
        win=th[sy:cy+w, sx:cx+w].copy(); inb=Bc[sy:cy+w, sx:cx+w]
        win[~inb]=0
        ks=LOWCUT.get((sid,key),5)
        cut=max(frac*(foil_n-band_n), med+ks*sig)
        reg=nd.binary_closing(win>cut,iterations=3)
        lab,n=nd.label(reg)
        best=None
        for i,sl in enumerate(nd.find_objects(lab),1):
            m=lab[sl]==i; a_=m.sum()
            yy,xx=np.where(m); my=yy.mean()+sl[0].start; mx=xx.mean()+sl[1].start
            dist=np.hypot(my-(cy-sy), mx-(cx-sx))/PPM
            if dist>4.5: continue
            sc=a_*float(win[sl][m].max())
            if best is None or sc>best[0]: best=(sc,i)
        if best is None:
            res[key]=dict(visible=False,d=0.0,h=0,w=0,peak=0,snr=0,y_px=cy+y0,x_px=cx+x0,y_mm=ym,x_mm=xm,comp=np.zeros((1,1),bool),sy=cy+y0,sx=cx+x0); continue
        comp=nd.binary_fill_holes(lab==best[1])
        pk=float(win[comp].max())
        area=comp.sum(); d=2*np.sqrt(area/np.pi)/PPM
        yy,xx=np.where(comp); hcy,hcx=yy.mean()+sy, xx.mean()+sx
        snr=(pk-med)/sig
        res[key]=dict(visible=bool(d>=0.5 and snr>=10), d=round(d,2), h=round((np.ptp(yy)+1)/PPM,2), w=round((np.ptp(xx)+1)/PPM,2),
                      peak=round(pk,3), snr=round(float(snr),1),
                      y_px=int(hcy+y0), x_px=int(hcx+x0), y_mm=round(hcy/PPM,1), x_mm=round(hcx/PPM,1),
                      comp=comp, sy=sy+y0, sx=sx+x0)
    return res, dict(a=a,wg=wg,sig=sig,med=med)
