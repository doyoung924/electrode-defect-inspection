from common import *
from scipy import ndimage as nd
from PIL import ImageDraw
S=4  # downscale factor for segmentation
def gray_small(sid):
    im=load(ALL[sid]).convert('RGB')
    a=np.asarray(im).astype(np.float32)
    sm=a.reshape(a.shape[0]//S,S,a.shape[1]//S,S,3).mean((1,3))
    return a, sm
def band_mask(sm, thr=None, open_it=4):
    g=sm.mean(2); H,W=g.shape
    c=g[int(H*0.15):int(H*0.7), int(W*0.25):int(W*0.65)]
    band_lv=np.percentile(c,20)
    board_lv=np.median(g[:int(H*0.04),:])
    if thr is None: thr=band_lv+0.5*(board_lv-band_lv)
    m=nd.binary_opening(g<thr,iterations=open_it)
    lab,n=nd.label(m)
    box=np.zeros_like(m); box[int(H*0.25):int(H*0.55), int(W*0.3):int(W*0.6)]=1
    counts=nd.sum(box,lab,range(1,n+1))
    i=int(np.argmax(counts))+1
    bm=nd.binary_dilation(lab==i,iterations=open_it)&(g<thr)
    bm=nd.binary_fill_holes(bm)
    bm=nd.binary_opening(bm,iterations=2)
    lab2,n2=nd.label(bm)
    if n2>1:
        sz=nd.sum(bm,lab2,range(1,n2+1)); bm=lab2==(np.argmax(sz)+1)
    core=bm
    from skimage.morphology import convex_hull_image
    hull=nd.binary_dilation(convex_hull_image(core),iterations=3)
    ring=nd.binary_dilation(core,iterations=40)&~nd.binary_dilation(core,iterations=15)
    foil_lv=np.median(g[ring])
    thr_hi=band_lv+0.5*(foil_lv-band_lv)
    out=(g<thr_hi)&hull
    out=nd.binary_opening(out,iterations=2)
    lab3,n3=nd.label(out); sz=nd.sum(out,lab3,range(1,n3+1)); out=lab3==(np.argmax(sz)+1)
    out=nd.binary_fill_holes(out)
    return out, (round(float(thr),1), round(float(thr_hi),1), round(float(band_lv),1), round(float(foil_lv),1))
def paper(sm):
    g=sm.mean(2); H,W=g.shape
    sub=g[int(H*0.7):, :int(W*0.4)]
    m=sub>170
    m=nd.binary_opening(m,iterations=2)
    lab,n=nd.label(m); cands=[]
    for i in range(1,n+1):
        ys,xs=np.where(lab==i); h=np.ptp(ys)+1; w=np.ptp(xs)+1; a=len(ys)
        if a<500: continue
        sq=min(h,w)/max(h,w); fill=a/(h*w)
        cands.append((sq*fill*a, ys.min()+int(H*0.7), ys.max()+int(H*0.7), xs.min(), xs.max(), a, round(sq,2), round(fill,2)))
    cands.sort(reverse=True)
    return cands[:3]

def ruler_scale(a, bm):
    """ruler: bright vertical strip right of band; mm tick period via FFT on full-res column profile"""
    g=a.mean(2); H,W=g.shape
    ys,xs=np.where(bm); y0,y1=ys.min()*S, ys.max()*S; xr=xs.max()*S
    # tick texture: vertical gradient energy per column, right of band
    sub=g[y0:y1, xr+100:min(W,xr+1400)]
    gy=np.abs(np.diff(sub,axis=0)).mean(0)
    col=int(np.argmax(nd.uniform_filter1d(gy,40)))+xr+100
    best=None
    for dx in range(-150,151,10):
        x=col+dx
        p=g[y0:y1, x:x+8].mean(1); p=p-nd.uniform_filter1d(p,41)
        n=len(p); N=1<<17
        F=np.abs(np.fft.rfft(p*np.hanning(n),N)); fr=np.fft.rfftfreq(N)
        lo,hi=np.searchsorted(fr,1/13),np.searchsorted(fr,1/8)
        k=lo+int(np.argmax(F[lo:hi])); pk=F[k]/np.median(F[lo:hi])
        if best is None or pk>best[0]: best=(pk,1/fr[k],x)
    return round(best[1],3), round(best[0],1), best[2]

def ruler_scale2(a, bm, x_hint=None):
    g=a.mean(2); H,W=g.shape
    ys,xs=np.where(bm); y0,y1=ys.min()*S, ys.max()*S; xr=xs.max()*S
    sub=g[y0:y1, xr+100:min(W,xr+1400)]
    gy=np.abs(np.diff(sub,axis=0)).mean(0)
    col=int(np.argmax(nd.uniform_filter1d(gy,40)))+xr+100
    res=[]
    for dx in range(-120,121,8):
        x=col+dx
        if x<0 or x+6>W: continue
        p=g[y0:y1, x:x+6].mean(1); p=p-nd.uniform_filter1d(p,61)
        p=(p-p.mean())/(p.std()+1e-6)
        ac=np.correlate(p,p,'full')[len(p)-1:]/len(p)
        # 10mm lag region and 50mm lag region
        def peak(lo,hi):
            k=lo+int(np.argmax(ac[lo:hi]))
            # parabolic refine
            if 0<k<len(ac)-1:
                d=(ac[k-1]-ac[k+1])/(2*(ac[k-1]-2*ac[k]+ac[k+1])+1e-9); return k+d, ac[k]
            return k, ac[k]
        if len(ac)<1300: continue
        l10,v10=peak(90,125); l100,v100=peak(int(l10*10)-40,int(l10*10)+40)
        res.append((v100, l100/100, l10, x))
    res.sort(reverse=True)
    v,ppm,l10,x=res[0]
    return round(ppm,3), round(v,2), round(l10/10,3), x

PPM=10.45  # px/mm at full res (ruler FFT median, batch 1); same camera setup throughout
def roi_mask(bm, side_mm=5, end_mm=10):
    rx=int(round(side_mm*PPM/S)); ry=int(round(end_mm*PPM/S))
    r=nd.binary_erosion(bm, structure=np.ones((1,2*rx+1),bool))
    r=nd.binary_erosion(r, structure=np.ones((2*ry+1,1),bool))
    lab,n=nd.label(r)
    if n>1:
        sz=nd.sum(r,lab,range(1,n+1)); r=lab==(np.argmax(sz)+1)
    return r
def paper_white(sm):
    g=sm.mean(2); H,W=g.shape
    y0=int(H*0.7); sub=g[y0:,:int(W*0.45)]
    m=nd.binary_opening(sub>110,iterations=2)
    lab,n=nd.label(m); best=None
    for i,sl in enumerate(nd.find_objects(lab),1):
        h=sl[0].stop-sl[0].start; w=sl[1].stop-sl[1].start; a=(lab[sl]==i).sum()
        if a<2500: continue
        sq=min(h,w)/max(h,w); sc=sq*a/(h*w)
        if best is None or sc>best[0]: best=(sc,i,sl)
    pm=nd.binary_erosion(lab==best[1],iterations=3)
    full=np.zeros(g.shape,bool); full[y0:,:int(W*0.45)]=pm
    px=sm[full]
    return np.percentile(px,90,axis=0), full, best[2]

def paper_med(sm):
    _,pm,_=paper_white(sm)
    px=sm[pm]; sat=(px.max(1)>=250).mean()
    return np.median(px,axis=0), sat
def spots(a, roi, white_gray, open_px=21, thr=0.035, min_area=3):
    g=a.mean(2)/white_gray
    R=np.kron(roi,np.ones((S,S),bool))
    ys,xs=np.where(R); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    c=g[y0:y1,x0:x1]; r=R[y0:y1,x0:x1]
    th=c-nd.grey_opening(c,size=(open_px,open_px))
    m=(th>thr)&r
    lab,n=nd.label(m)
    if n==0: return [], (y0,x0), th
    sz=nd.sum(m,lab,range(1,n+1)); pk=nd.maximum(th,lab,range(1,n+1))
    com=nd.center_of_mass(m,lab,range(1,n+1))
    out=[(int(s),float(p),int(cy)+y0,int(cx)+x0) for s,p,(cy,cx) in zip(sz,pk,com) if s>=min_area]
    return out,(y0,x0),th

def analyze(sid, k=8, min_area=4, open_px=21):
    a,sm=gray_small(sid); bm,_=band_mask(sm); roi=roi_mask(bm)
    pmed,sat=paper_med(sm); wg=float(pmed.mean())
    _,(y0,x0),th=spots(a,roi,wg,thr=9,open_px=open_px)
    R=np.kron(roi,np.ones((S,S),bool))[y0:y0+th.shape[0],x0:x0+th.shape[1]]
    v=th[R]; med=np.median(v); sig=np.median(np.abs(v-med))*1.4826
    thr=med+k*sig
    m=(th>thr)&R
    lab,n=nd.label(m)
    sz=nd.sum(m,lab,range(1,n+1)) if n else []
    pk=nd.maximum(th,lab,range(1,n+1)) if n else []
    com=nd.center_of_mass(m,lab,range(1,n+1)) if n else []
    sp=[(int(s),float(p),int(cy)+y0,int(cx)+x0) for s,p,(cy,cx) in zip(sz,pk,com) if s>=min_area]
    area_cm2=roi.sum()*S*S/PPM**2/100
    return dict(a=a,sm=sm,bm=bm,roi=roi,white=wg,sat=sat,thr=thr,sig=sig,sp=sp,area=area_cm2,th=th,off=(y0,x0))
