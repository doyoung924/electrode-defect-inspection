"""
PatchCore 3-fold cross validation on hand-made coating specimens.
- normal A grade (17) split into 3 folds -> train on 2 folds, test on 1
- B grade (12), pinhole (11), foreign object (2), streak (3): always test
Outputs: results/<fold>/*.csv, results/summary.json, results/heatmaps/*.jpg
"""
import os, csv, json, argparse, time
import numpy as np
from PIL import Image
from scipy import ndimage as nd

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

PPM = 10.45          # px per mm (ruler, full-res originals)
STRIDE = 8           # feature cell size in px of the (possibly upscaled) crop
SCALE = 1.0          # input upscale factor (set by --scale)
def CELL(): return STRIDE / SCALE   # cell size in original px

# ---------------- data ----------------
def load_rgb(path):
    from pillow_heif import register_heif_opener
    register_heif_opener()
    return np.asarray(Image.open(path).convert('RGB')).astype(np.float32)

def load_mask(kit, sid, kind):
    m = np.asarray(Image.open(os.path.join(kit, 'masks', f'{sid}_{kind}.png'))) > 127
    return m

def prepare(kit, raw, row, margin=32):
    """returns normalized band crop (H,W,3 in 0..255), roi mask on crop (full res), crop origin"""
    sid = row['id']; s = int(row['mask_scale'])
    img = load_rgb(os.path.join(raw, row['file']))
    paper = np.array([float(row['paper_r']), float(row['paper_g']), float(row['paper_b'])], np.float32)
    img = np.clip(img / paper * 200.0, 0, 255)                     # white-paper normalization
    band = np.kron(load_mask(kit, sid, 'band'), np.ones((s, s), bool))
    roi = np.kron(load_mask(kit, sid, 'roi'), np.ones((s, s), bool))
    ys, xs = np.where(band)
    y0 = max(0, ys.min() - margin); y1 = min(img.shape[0], ys.max() + margin)
    x0 = max(0, xs.min() - margin); x1 = min(img.shape[1], xs.max() + margin)
    # make size divisible by 16
    y1 = y0 + ((y1 - y0) // 16) * 16; x1 = x0 + ((x1 - x0) // 16) * 16
    crop, rc = img[y0:y1, x0:x1], roi[y0:y1, x0:x1]
    if SCALE != 1.0:
        H, W = int(crop.shape[0] * SCALE), int(crop.shape[1] * SCALE)
        crop = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize((W, H), Image.BICUBIC)).astype(np.float32)
        rc = np.asarray(Image.fromarray(rc.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
    return crop, rc, (y0, x0)

def cell_mask(roi_crop):
    """ROI at feature resolution: cell is valid if its center pixel lies in ROI"""
    c = STRIDE // 2
    return roi_crop[c::STRIDE, c::STRIDE]

# ---------------- features ----------------
class WRN50Features:
    def __init__(self, device, dim=384, seed=0):
        import torchvision
        self.device = device
        m = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        m.eval().to(device)
        self.m = m
        g = torch.Generator().manual_seed(seed)
        self.proj = (torch.randn(512 + 1024, dim, generator=g) / np.sqrt(dim)).to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def __call__(self, crop):
        with torch.no_grad():
            return self._forward(crop)

    def _forward(self, crop):
        x = torch.from_numpy(crop / 255.0).permute(2, 0, 1)[None].float().to(self.device)
        x = (x - self.mean) / self.std
        m = self.m
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        f1 = m.layer1(x); f2 = m.layer2(f1); f3 = m.layer3(f2)
        f2 = torch.nn.functional.avg_pool2d(f2, 3, 1, 1)
        f3 = torch.nn.functional.avg_pool2d(f3, 3, 1, 1)
        f3 = torch.nn.functional.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        f = torch.cat([f2, f3], 1)[0].permute(1, 2, 0)            # H/8, W/8, 1536
        f = f @ self.proj
        return f.cpu().numpy().astype(np.float32)

class MockFeatures:
    """numpy stand-in for pipeline testing (no torch): local color stats per cell"""
    def __init__(self, dim=24, seed=0):
        self.proj = np.random.default_rng(seed).normal(size=(12, dim)).astype(np.float32)
    def __call__(self, crop):
        H, W = crop.shape[0] // STRIDE, crop.shape[1] // STRIDE
        c = crop[:H * STRIDE, :W * STRIDE].reshape(H, STRIDE, W, STRIDE, 3)
        mean = c.mean((1, 3)) / 255; mx = c.max((1, 3)) / 255; sd = c.std((1, 3)) / 255
        loc = nd.uniform_filter(mean, size=(5, 5, 1))
        f = np.concatenate([mean, mx, sd, mean - loc], -1)
        return f @ self.proj

# ---------------- kNN / coreset ----------------
def nn_dist(Q, B, B_img=None, q_img=None, device=None, chunk=None):
    """min L2 distance from each row of Q to rows of B (optionally excluding same image id)"""
    if chunk is None: chunk = max(256, int(1.5e8 // max(1, len(B))))
    Q = Q.astype(np.float32); B = B.astype(np.float32)
    if HAS_TORCH and device is not None:
        Bt = torch.from_numpy(B).to(device); bn = (Bt * Bt).sum(1)
        ex = None if B_img is None else torch.from_numpy(B_img == q_img).to(device)
        out = []
        for i in range(0, len(Q), chunk):
            q = torch.from_numpy(Q[i:i + chunk]).to(device)
            d = (q * q).sum(1, keepdim=True) - 2 * q @ Bt.T + bn[None]
            if ex is not None: d[:, ex] = float('inf')
            out.append(d.min(1).values.clamp_min(0).sqrt().cpu().numpy())
        return np.concatenate(out)
    bn = (B * B).sum(1); out = []
    ex = None if B_img is None else (B_img == q_img)
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        d = (q * q).sum(1, keepdims=True) - 2 * q @ B.T + bn[None]
        if ex is not None: d[:, ex] = np.inf
        out.append(np.sqrt(np.maximum(d.min(1), 0)))
    return np.concatenate(out)

def coreset(X, ratio, device=None, seed=0, proj_dim=128):
    n = max(1, int(len(X) * ratio))
    rng = np.random.default_rng(seed)
    if not (HAS_TORCH and device is not None):
        return rng.choice(len(X), n, replace=False)               # random subsample on CPU test path
    Xt = torch.from_numpy(X).to(device)
    P = torch.from_numpy(rng.normal(size=(X.shape[1], proj_dim)).astype(np.float32)).to(device)
    Z = Xt @ P
    idx = [int(rng.integers(len(X)))]
    d = ((Z - Z[idx[0]]) ** 2).sum(1)
    for _ in range(n - 1):
        i = int(torch.argmax(d)); idx.append(i)
        d = torch.minimum(d, ((Z - Z[i]) ** 2).sum(1))
    return np.array(idx)

# ---------------- evaluation helpers ----------------
def hole_hits(score, valid, origin, holes, thr):
    """per hole: max score within radius (d/2 + 1mm) of center"""
    y0, x0 = origin; H, W = score.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cy = y0 + (yy * STRIDE + STRIDE / 2) / SCALE; cx = x0 + (xx * STRIDE + STRIDE / 2) / SCALE
    out = []
    for h in holes:
        r = (float(h['지름_mm']) / 2 + 1.0) * PPM
        m = (cy - float(h['중심y_px'])) ** 2 + (cx - float(h['중심x_px'])) ** 2 <= r * r
        s = float(score[m].max()) if m.any() else float('nan')
        out.append(dict(hole=h['구멍'], d=float(h['지름_mm']), flag=h['판정'], max_score=s, detected=bool(s > thr)))
    return out

def away_from_holes(score, valid, origin, holes, gap_mm=5.0):
    y0, x0 = origin; H, W = score.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cy = y0 + (yy * STRIDE + STRIDE / 2) / SCALE; cx = x0 + (xx * STRIDE + STRIDE / 2) / SCALE
    keep = valid.copy()
    for h in holes:
        r = (float(h['지름_mm']) / 2 + gap_mm) * PPM
        keep &= (cy - float(h['중심y_px'])) ** 2 + (cx - float(h['중심x_px'])) ** 2 > r * r
    return keep

def n_regions(mask):
    return int(nd.label(mask)[1])

def auroc(neg, pos):
    neg = np.asarray(neg); pos = np.asarray(pos)
    if len(neg) == 0 or len(pos) == 0: return float('nan')
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))

def save_heatmap(path, crop, score, valid, thr, holes=None, origin=None):
    s = np.where(valid, score, 0)
    up = np.kron(s, np.ones((STRIDE, STRIDE)))[:crop.shape[0], :crop.shape[1]]
    v = np.clip(up / (2 * thr), 0, 1)
    base = np.clip(crop * 2.5, 0, 255) * 0.6
    heat = np.stack([v * 255, (1 - abs(v - 0.5) * 2) * 120, (1 - v) * 40], -1)
    img = np.clip(base + heat * 0.6, 0, 255)
    hot = np.kron((s > thr), np.ones((STRIDE, STRIDE), bool))[:crop.shape[0], :crop.shape[1]]
    edge = hot ^ nd.binary_erosion(hot, iterations=2)
    img[edge] = [0, 255, 255]
    im = Image.fromarray(img.astype(np.uint8))
    if holes:
        from PIL import ImageDraw
        d = ImageDraw.Draw(im); y0, x0 = origin
        for h in holes:
            cy = (float(h['중심y_px']) - y0) * SCALE; cx = (float(h['중심x_px']) - x0) * SCALE; r = (float(h['지름_mm']) / 2 + 2) * PPM * SCALE
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255), width=4)
    k = int(3 * SCALE); im = im.resize((im.size[0] // k, im.size[1] // k))
    im.save(path, quality=85)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kit', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mock', action='store_true', help='numpy features, for pipeline test only')
    ap.add_argument('--coreset', type=float, default=0.1)
    ap.add_argument('--only', default='', help='comma ids to restrict (debug)')
    ap.add_argument('--scale', type=float, default=1.0, help='upscale input crop (2 = 2x)')
    ap.add_argument('--exclude-train', default='', help='comma ids: A specimens moved out of training into test')
    a = ap.parse_args()
    global SCALE
    SCALE = a.scale
    os.makedirs(os.path.join(a.out, 'heatmaps'), exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(a.kit, 'specimens.csv'), encoding='utf-8')))
    if a.only: rows = [r for r in rows if r['id'] in a.only.split(',')]
    excl = [x for x in a.exclude_train.split(',') if x]
    for r in rows:
        if r['id'] in excl: r['role'] = 'A_excluded'
    holes = {}
    for h in csv.DictReader(open(os.path.join(a.kit, 'holes.csv'), encoding='utf-8-sig')):
        holes.setdefault(h['시편'], []).append(h)

    device = None
    if not a.mock:
        assert HAS_TORCH, 'torch not installed'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('device', device)
        fx = WRN50Features(device)
    else:
        fx = MockFeatures()

    # 1) features for every specimen (computed once)
    feats = {}; t0 = time.time()
    for r in rows:
        crop, roi, origin = prepare(a.kit, a.raw, r)
        f = fx(crop); valid = cell_mask(roi)[:f.shape[0], :f.shape[1]]
        f = f[:valid.shape[0], :valid.shape[1]]
        feats[r['id']] = dict(f=f.astype(np.float16), valid=valid, origin=origin, crop_shape=crop.shape, row=r)
        del crop
        print(f"features {r['id']} {f.shape} valid {int(valid.sum())}  {time.time()-t0:.0f}s", flush=True)

    folds = sorted({r['role'] for r in rows if r['role'].startswith('A_fold')})
    summary = dict(folds={}, config=dict(coreset=a.coreset, mock=a.mock, ppm=PPM, stride=STRIDE, scale=SCALE, exclude_train=excl))
    for fold in folds:
        tr = [i for i, v in feats.items() if v['row']['role'].startswith('A_fold') and v['row']['role'] != fold]
        te = [i for i, v in feats.items() if not (v['row']['role'].startswith('A_fold') and v['row']['role'] != fold)]
        X = np.concatenate([feats[i]['f'][feats[i]['valid']] for i in tr]).astype(np.float32)
        Xid = np.concatenate([np.full(int(feats[i]['valid'].sum()), k) for k, i in enumerate(tr)])
        sel = coreset(X, a.coreset, device); B, Bid = X[sel], Xid[sel]
        print(f'{fold}: train {len(tr)} imgs, {len(X)} feats -> bank {len(B)}', flush=True)
        # threshold: leave-one-image-out max score over training images
        loo, pooled = {}, []
        for k, i in enumerate(tr):
            q = feats[i]['f'][feats[i]['valid']]
            dd = nn_dist(q, B, Bid, k, device); loo[i] = float(dd.max()); pooled.append(dd)
        pooled = np.concatenate(pooled)
        THRS = {'strict': max(loo.values()), 'p999': float(np.percentile(pooled, 99.9))}
        thr = THRS['strict']
        print(f'{fold}: thr strict {THRS["strict"]:.4f} (set by {max(loo, key=loo.get)}), p99.9 {THRS["p999"]:.4f}', flush=True)
        fd = os.path.join(a.out, fold); os.makedirs(fd, exist_ok=True)
        img_rows, hole_rows = [], []
        for i in te:
            v = feats[i]; r = v['row']
            sc = np.zeros(v['valid'].shape, np.float32)
            sc[v['valid']] = nn_dist(v['f'][v['valid']], B, device=device)
            np.save(os.path.join(fd, f'{i}_score.npy'), sc)
            kind = r['kind']; grade = r['grade']
            rec = dict(id=i, kind=kind, grade=grade, role=r['role'], max_score=float(sc[v['valid']].max()),
                       roi_cm2=round(float(v['valid'].sum()) * (CELL() / PPM) ** 2 / 100, 1))
            for tn, tv in THRS.items():
                rec[f'over_{tn}'] = bool(rec['max_score'] > tv)
                rec[f'regions_{tn}'] = n_regions((sc > tv) & v['valid'])
            if kind == '핀홀':
                hh = hole_hits(sc, v['valid'], v['origin'], holes.get(i, []), thr)
                for x in hh:
                    x.update(id=i); x.pop('detected')
                    for tn, tv in THRS.items(): x[f'det_{tn}'] = bool(x['max_score'] > tv)
                    hole_rows.append(x)
                keep = away_from_holes(sc, v['valid'], v['origin'], holes.get(i, []))
                rec['bg_max_score'] = float(sc[keep].max()) if keep.any() else float('nan')
                for tn, tv in THRS.items(): rec[f'bg_regions_{tn}'] = n_regions((sc > tv) & keep)
            img_rows.append(rec)
            if fold == folds[0] or kind == '핀홀' or r['role'] in (fold, 'A_excluded'):
                crop, roi, origin = prepare(a.kit, a.raw, r)
                save_heatmap(os.path.join(a.out, 'heatmaps', f'{fold}_{i}.jpg'), crop, sc, v['valid'], thr,
                             holes.get(i) if kind == '핀홀' else None, origin)
        for name, rr in (('images', img_rows), ('holes', hole_rows)):
            if rr:
                with open(os.path.join(fd, f'{name}.csv'), 'w', newline='', encoding='utf-8-sig') as fh:
                    keys = list(dict.fromkeys(k for x in rr for k in x.keys()))
                    w = csv.DictWriter(fh, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rr)
        # fold summary
        def frac(xs): return [int(sum(xs)), len(xs)]
        normA = [x for x in img_rows if x['kind'] == '정상' and x['grade'] == 'A' and x['role'] != 'A_excluded']
        glare = [x for x in img_rows if x['role'] == 'A_excluded']
        normB = [x for x in img_rows if x['kind'] == '정상' and x['grade'] == 'B']
        pin = [x for x in img_rows if x['kind'] == '핀홀']
        evh = [x for x in hole_rows if x['flag'] == '평가']
        bins = {'1-1.5': (1, 1.5), '1.5-2': (1.5, 2), '2-2.5': (2, 2.5), '2.5-3': (2.5, 3), '3+': (3, 99)}
        neg = [x['max_score'] for x in normA + normB + glare]
        posimg = [x['max_score'] for x in img_rows if x['kind'] != '정상']
        per = {}
        for tn in THRS:
            per[tn] = dict(
                threshold=THRS[tn],
                false_alarm_A=frac([x[f'over_{tn}'] for x in normA]),
                false_alarm_B=frac([x[f'over_{tn}'] for x in normB]),
                excluded_A_over={x['id']: x[f'over_{tn}'] for x in glare},
                pinhole_background_alarm_imgs=frac([x[f'bg_regions_{tn}'] > 0 for x in pin]),
                holes_detected=frac([x[f'det_{tn}'] for x in evh]),
                holes_by_size={k: frac([x[f'det_{tn}'] for x in evh if lo <= x['d'] < hi]) for k, (lo, hi) in bins.items()},
                small_holes_detected=frac([x[f'det_{tn}'] for x in hole_rows if x['flag'] != '평가']),
                defect_imgs_over_thr={x['id']: x[f'over_{tn}'] for x in img_rows if x['kind'] in ('이물', '줄무늬')})
        summary['folds'][fold] = dict(train=tr, loo_max=loo, image_auroc_normal_vs_defect=auroc(neg, posimg), **per)
        print(json.dumps(summary['folds'][fold], ensure_ascii=False, default=str)[:600], flush=True)
    # aggregate over folds
    agg = {}
    for tn in ('strict', 'p999'):
        F = [summary['folds'][f][tn] for f in folds]
        def add(key): return [sum(x[key][0] for x in F), sum(x[key][1] for x in F)]
        agg[tn] = dict(
            false_alarm_A=add('false_alarm_A'), false_alarm_B=add('false_alarm_B'),
            pinhole_background_alarm_imgs=add('pinhole_background_alarm_imgs'),
            holes_detected=add('holes_detected'), small_holes_detected=add('small_holes_detected'),
            holes_by_size={k: [sum(x['holes_by_size'][k][0] for x in F), sum(x['holes_by_size'][k][1] for x in F)] for k in F[0]['holes_by_size']},
            defect_imgs_over_thr={k: sum(x['defect_imgs_over_thr'][k] for x in F) for k in F[0]['defect_imgs_over_thr']},
            excluded_A_over={k: sum(x['excluded_A_over'][k] for x in F) for k in F[0]['excluded_A_over']})
    agg['image_auroc_mean'] = float(np.nanmean([summary['folds'][f]['image_auroc_normal_vs_defect'] for f in folds]))
    summary['aggregate'] = agg
    print('\n=== AGGREGATE (counts summed over folds: [hit, total]) ===')
    print(json.dumps(agg, ensure_ascii=False, indent=1))
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print('done', time.time() - t0)

if __name__ == '__main__':
    main()
