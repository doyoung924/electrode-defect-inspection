from seg import *
def find_holes(sid, open_px=81, k=10, min_d=0.5):
    a,sm=gray_small(sid); bm,_=band_mask(sm)
    pmed,sat=paper_med(sm); wg=float(pmed.mean())
    B=np.kron(bm,np.ones((S,S),bool)); Bi=nd.binary_erosion(B,iterations=20)
    ys,xs=np.where(B); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    g=a.mean(2)/wg; c=g[y0:y1,x0:x1]; r=Bi[y0:y1,x0:x1]
    th=c-nd.grey_opening(c,size=(open_px,open_px))
    v=th[r]; med=np.median(v); sig=np.median(np.abs(v-med))*1.4826
    thr=max(med+k*sig,0.05)
    m=nd.binary_fill_holes(nd.binary_closing((th>thr)&r,iterations=2))
    lab,n=nd.label(m); out=[]
    for i,sl in enumerate(nd.find_objects(lab),1):
        mm=lab[sl]==i; area=mm.sum(); d=2*np.sqrt(area/np.pi)/PPM
        if d<min_d: continue
        cy,cx=nd.center_of_mass(mm); cy+=sl[0].start; cx+=sl[1].start
        pk=th[sl][mm].max(); h=(sl[0].stop-sl[0].start)/PPM; w=(sl[1].stop-sl[1].start)/PPM
        out.append(dict(d=round(d,2),h=round(h,2),w=round(w,2),peak=round(float(pk),3),
            y_mm=round((cy)/PPM,1), x_mm=round((cx)/PPM,1), cy=int(cy+y0), cx=int(cx+x0)))
    out.sort(key=lambda o:o['y_mm'])
    bw=(x1-x0)/PPM
    return out, dict(thr=thr,sig=sig,band_w=bw,band_top=y0,band_left=x0,a=a,wg=wg,B=B)
