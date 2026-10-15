import numpy as np, csv, pickle, json
from PIL import Image
KIT='/home/claude/spec/kit'; PPM=10.45; SCALE=2.0; STRIDE=8; H0,W0=4032,3024
rows={r['id']:r for r in csv.DictReader(open(f'{KIT}/specimens.csv',encoding='utf-8'))}
def mask(sid,k): return np.asarray(Image.open(f'{KIT}/masks/{sid}_{k}.png'))>127
def geom(sid, margin=32):
    s=4
    band=mask(sid,'band'); ys,xs=np.where(band); ys=ys*s; xs=xs*s
    y0=max(0,ys.min()-margin); y1=min(H0,ys.max()+s-1+margin)   # kron upsample: last row index = ymax*4+3
    x0=max(0,xs.min()-margin); x1=min(W0,xs.max()+s-1+margin)
    y1=y0+((y1-y0)//16)*16; x1=x0+((x1-x0)//16)*16
    Hc,Wc=int((y1-y0)*SCALE),int((x1-x0)*SCALE)
    return y0,x0,Hc//STRIDE,Wc//STRIDE
def cell_centers(sid,shape):
    y0,x0,_,_=geom(sid); h,w=shape
    yy,xx=np.mgrid[0:h,0:w]
    return y0+(yy*STRIDE+STRIDE/2)/SCALE, x0+(xx*STRIDE+STRIDE/2)/SCALE
def valid_cells(sid,shape):
    cy,cx=cell_centers(sid,shape)
    roi=np.kron(mask(sid,'roi'),np.ones((4,4),bool))
    # prepare(): roi crop upscaled with NEAREST, then sampled at c::STRIDE -> approx original pixel at floor(center)
    iy=np.clip(np.floor(cy).astype(int),0,H0-1); ix=np.clip(np.floor(cx).astype(int),0,W0-1)
    return roi[iy,ix]
