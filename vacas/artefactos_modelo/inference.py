import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import timm
from xgboost import XGBRegressor
from ultralytics.models.yolo import YOLO

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------
# RUTAS
# -----------------------------------
NEW_IMG_DIR = Path('/content/drive/MyDrive/Tesis/TESIS-VACAS/Pruebas')
EXPERIMENT_DIR = Path('/content/drive/MyDrive/Tesis/TESIS-VACAS/resultados/WRN50_2_XGB_v1')

OUT_ROOT = Path('/content/_PRED_BASELINE')
CONT_DIR = OUT_ROOT / 'contorno'
SIL_DIR = OUT_ROOT / 'silueta'
PRED_DIR = OUT_ROOT / 'pred'
for p in [CONT_DIR, SIL_DIR, PRED_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# CONFIGURACIÓN BASELINE
# -----------------------------------
FIXED_CANVAS_WH = (1280, 960)
CONF = 0.20  # BASELINE SINGLE CONFIG
IOU = 0.45
SIL_SIZE = (256, 256)
EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

# Modelos
COW_YOLO_PATH = 'yolov8x-seg.pt'
CIRCLE_MODEL_PATH = '/content/drive/MyDrive/Tesis/TESIS-VACAS/last_actual.pt'

cow_model = YOLO(COW_YOLO_PATH)
circle_model = YOLO(CIRCLE_MODEL_PATH) if Path(CIRCLE_MODEL_PATH).exists() else None

# -----------------------------------
# UTILIDADES
# -----------------------------------
def rotate_if_vertical(bgr):
    return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE) if bgr.shape[0] > bgr.shape[1] else bgr

def resize_pad_fixed(img_bgr, target_wh):
    H, W = img_bgr.shape[:2]
    tw, th = target_wh
    r = min(tw / W, th / H)
    nw, nh = int(round(W * r)), int(round(H * r))
    img_rs = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_CUBIC)
    canvas = np.full((th, tw, 3), 255, dtype=np.uint8)
    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = img_rs
    return canvas

def detect_circle_in_canvas(bgr_fixed):
    H, W = bgr_fixed.shape[:2]
    if circle_model is not None:
        r = circle_model.predict(bgr_fixed[:, :, ::-1], imgsz=min(max(W,H), 960), conf=0.20,
                                 device=0 if torch.cuda.is_available() else 'cpu',
                                 verbose=False, retina_masks=True)[0]
        try:
            if r.masks is not None and len(r.masks.data) > 0:
                pts = r.masks.xy[0]
                pts = np.asarray(pts, dtype=np.float32)
                (cx, cy), rr = cv2.minEnclosingCircle(pts)
                return (int(round(cx)), int(round(cy)), int(round(rr)))
            if r.boxes is not None and len(r.boxes) > 0:
                x1, y1, x2, y2 = r.boxes.xyxy[0].cpu().numpy().tolist()
                cx = 0.5*(x1+x2); cy = 0.5*(y1+y2); rr = 0.5*max((x2-x1),(y2-y1))
                return (int(round(cx)), int(round(cy)), int(round(rr)))
        except Exception:
            pass
    return None

def keep_largest_component(bin_mask):
    m = (bin_mask > 0).astype(np.uint8)
    num, lab = cv2.connectedComponents(m)
    if num <= 1: return m*255
    best, area = 0, -1
    for lb in range(1, num):
        a = int((lab==lb).sum())
        if a > area:
            area, best = a, lb
    return ((lab==best).astype(np.uint8))*255

def postprocess_mask(mask, H, W):
    """CALIBRADO PUNTO MEDIO: Coincidir exactamente con entrenamiento"""
    m = (mask>0).astype(np.uint8)*255
    
    # Dilate moderado (punto medio entre 0.003 y 0.006)
    k_dilate = max(3, int(0.0045*min(H,W)))  # ~4-5 px
    K_dilate = np.ones((k_dilate, k_dilate), np.uint8)
    m = cv2.dilate(m, K_dilate, iterations=1)
    
    # Morfología moderada
    k = max(4, int(0.009*min(H,W)))  # ~8-9 px (punto medio entre 0.006 y 0.012)
    K = np.ones((k,k), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K, iterations=2)  # 2 en vez de 1 o 3
    # NO hacer MORPH_OPEN (preservar área)
    
    # Fill holes
    mm = m.copy(); ff = np.zeros((H+2, W+2), np.uint8)
    cv2.floodFill(mm, ff, (0,0), 255)
    holes = cv2.bitwise_not(mm)
    m = cv2.bitwise_or(m, holes)
    
    # Componente mayor
    m = keep_largest_component(m)
    
    # Smooth moderado (convex hull parcial)
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        big = max(cnts, key=cv2.contourArea)
        # Híbrido: approxPolyDP con epsilon un poco más grande
        eps = 0.002 * cv2.arcLength(big, True)  # Doble que antes, mitad que convexHull
        approx = cv2.approxPolyDP(big, eps, True)
        m[:] = 0
        cv2.drawContours(m, [approx], -1, 255, thickness=cv2.FILLED)
    
    return m

def mask_from_bbox(H, W, box):
    m = np.zeros((H, W), np.uint8)
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W-1, x2), min(H-1, y2)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = 255
    return m

def cheap_fallback_mask(fixed_bgr):
    H, W = fixed_bgr.shape[:2]
    lab = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    L = cv2.GaussianBlur(L, (5,5), 0)
    _, th = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() > 127:
        th = 255 - th
    K = np.ones((5,5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, K, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  K, iterations=1)
    return keep_largest_component(th)

def morph_from_silhouette(path_sil: Path):
    def _nanrow():
        return dict(area_px=np.nan, perim_px=np.nan, bbox_w=np.nan, bbox_h=np.nan,
                    aspect=np.nan, hu1=np.nan, hu2=np.nan, hu3=np.nan, fill_frac=np.nan)
    img = cv2.imread(str(path_sil))
    if img is None: return _nanrow()
    H,W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0,50,50), (10,255,255))
    mask2 = cv2.inRange(hsv, (170,50,50), (180,255,255))
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    mask = (mask>0).astype(np.uint8)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return dict(area_px=0.0, perim_px=0.0, bbox_w=0.0, bbox_h=0.0,
                    aspect=0.0, hu1=0.0, hu2=0.0, hu3=0.0, fill_frac=0.0)
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perim = float(cv2.arcLength(c, True))
    x,y,w,h = cv2.boundingRect(c)
    aspect = (w/h) if h>0 else 0.0
    hu = cv2.HuMoments(cv2.moments(c)).flatten()[:3]
    hu = np.sign(hu)*np.log10(np.abs(hu)+1e-12)
    fill_frac = area/float(H*W)
    return dict(area_px=area, perim_px=perim, bbox_w=float(w), bbox_h=float(h),
                aspect=float(aspect), hu1=float(hu[0]), hu2=float(hu[1]), hu3=float(hu[2]),
                fill_frac=float(fill_frac))

# -----------------------------------
# PREPROCESAR IMÁGENES RAW
# -----------------------------------
print("="*80)
print("🔧 BASELINE SIMPLE - Preprocesando imágenes RAW")
print("="*80)

raw_imgs = [p for p in NEW_IMG_DIR.rglob('*') if p.suffix.lower() in EXTS]
if not raw_imgs:
    raise RuntimeError(f"No se encontraron imágenes en {NEW_IMG_DIR}")

print(f"\n📊 Procesando {len(raw_imgs)} imágenes...")

saved_ok = 0
for src in tqdm(raw_imgs, desc="Procesando"):
    try:
        bgr0 = cv2.imread(str(src))
        if bgr0 is None:
            continue

        # 1. Normalizar canvas
        fixed = resize_pad_fixed(rotate_if_vertical(bgr0), FIXED_CANVAS_WH)
        H, W = fixed.shape[:2]

        # 2. Detectar círculo
        cc = detect_circle_in_canvas(fixed)

        # 3. Segmentar con YOLO BASELINE (CONF=0.20, single config)
        rgb = cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB)
        cow_model.to(DEVICE)
        r = cow_model.predict(source=rgb, conf=CONF, iou=IOU, imgsz=1280, verbose=False)[0]

        best = None
        if r is not None:
            names = getattr(r, "names", None)
            cow_ids = None
            if isinstance(names, dict):
                cow_ids = [k for k,v in names.items() if str(v).lower() == "cow"]

            if r.masks is not None and len(r.masks.data) > 0:
                raw = r.masks.data.cpu().numpy().astype(np.uint8)
                masks = [cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) for m in raw]
                keep = list(range(len(masks)))

                if getattr(r, "boxes", None) is not None and len(r.boxes) == len(masks) and cow_ids:
                    cls = r.boxes.cls.cpu().numpy().astype(int).tolist()
                    keep = [i for i,c in enumerate(cls) if c in cow_ids] or keep

                confs = r.boxes.conf.cpu().numpy().tolist() if getattr(r,"boxes",None) else [0.0]*len(masks)
                cand = [(i, (m>0).sum(), confs[i]) for i,m in enumerate(masks) if i in keep]

                if cand:
                    i = max(cand, key=lambda t: (t[1], t[2]))[0]
                    best = (masks[i].astype(np.uint8) * 255)

            if best is None and getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
                areas = (r.boxes.xyxy[:,2] - r.boxes.xyxy[:,0]) * (r.boxes.xyxy[:,3] - r.boxes.xyxy[:,1])
                i = int(torch.argmax(areas).item())
                best = mask_from_bbox(H, W, r.boxes.xyxy[i].cpu().numpy().tolist())

        if best is None or best.sum() < 1000:
            best = cheap_fallback_mask(fixed)

        # 4. Postprocesar máscara (morfología baseline)
        best = postprocess_mask(best, H, W)

        # 5. Guardar contorno
        stem = src.stem
        cont = fixed.copy()
        cnts,_ = cv2.findContours(best, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        smooth = []
        for c in cnts:
            eps = 0.003*cv2.arcLength(c, True)
            smooth.append(cv2.approxPolyDP(c, eps, True))
        if smooth:
            cv2.drawContours(cont, smooth, -1, (0,255,0), thickness=2)
        if cc is not None:
            cx,cy,rr = cc
            cv2.circle(cont, (cx,cy), rr, (255,0,255), 2)

        cont_p = CONT_DIR / f"{stem}_contorno.jpg"
        cv2.imwrite(str(cont_p), cont)

        # 6. Guardar silueta
        sil = np.full((H,W,3), 255, np.uint8)
        sil[best>0] = (0,0,255)
        sil = cv2.resize(sil, SIL_SIZE, interpolation=cv2.INTER_NEAREST)

        sil_p = SIL_DIR / f"{stem}_sil.jpg"
        cv2.imwrite(str(sil_p), sil)

        saved_ok += 1

    except Exception as e:
        print(f"\n⚠️ Error: {e}")

print(f"\n✅ Imágenes procesadas: {saved_ok}/{len(raw_imgs)}")

# -----------------------------------
# CONSTRUIR DATAFRAME
# -----------------------------------
cont_files = [p for p in CONT_DIR.rglob('*') if p.suffix.lower() in EXTS and '_contorno' in p.stem.lower()]
sil_files = [p for p in SIL_DIR.rglob('*') if p.suffix.lower() in EXTS and '_sil' in p.stem.lower()]

sil_map = {Path(p).stem.replace('_sil', ''): p for p in sil_files}

records = []
for p_cont in cont_files:
    base_name = Path(p_cont).stem.replace('_contorno', '')
    p_sil = sil_map.get(base_name, None)
    sample_id = base_name

    morph_feats = morph_from_silhouette(p_sil) if p_sil else {}

    record = {
        'sample_id': sample_id,
        'contorno_path': str(p_cont),
        'sil_path': str(p_sil) if p_sil else ''
    }
    record.update(morph_feats)
    records.append(record)

df_new = pd.DataFrame(records)
print(f"\n📊 DataFrame creado: {len(df_new)} muestras")

# -----------------------------------
# BACKBONE Y EMBEDDINGS
# -----------------------------------
manifest, _ = json.loads((EXPERIMENT_DIR / "inference_manifest.json").read_text()), None

TIMM_MODEL = manifest.get('timm_model', 'wide_resnet50_2')
IMG_SIZE = int(manifest.get('img_size', 224))
IMAGENET_MEAN = manifest.get('imagenet_mean', [0.485, 0.456, 0.406])
IMAGENET_STD = manifest.get('imagenet_std', [0.229, 0.224, 0.225])
MORPH_COLS = manifest.get('morph_cols', ['area_px','perim_px','bbox_w','bbox_h','aspect','hu1','hu2','hu3','fill_frac'])

# -----------------------------------
# DIAGNÓSTICO: FEATURES MORFOLÓGICAS
# -----------------------------------
print("\n" + "="*80)
print("🔬 DIAGNÓSTICO: FEATURES MORFOLÓGICAS EXTRAÍDAS")
print("="*80)

morph_stats = df_new[MORPH_COLS].describe()
print("\n📊 Estadísticas de features morfológicas:")
print(morph_stats.to_string())

# Verificar si hay valores anómalos
print("\n⚠️ Verificando valores anómalos:")
for col in MORPH_COLS:
    nan_count = df_new[col].isna().sum()
    zero_count = (df_new[col] == 0).sum()
    if nan_count > 0:
        print(f"   - {col}: {nan_count} NaN")
    if zero_count > 0:
        print(f"   - {col}: {zero_count} ceros")

# Mostrar samples individuales
print("\n📋 Samples individuales (primeras 5):")
print(df_new[['sample_id'] + MORPH_COLS].head().to_string(index=False))

# -----------------------------------
# BACKBONE
# -----------------------------------

class BackboneTIMM(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        m = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = m
        self.out_dim = getattr(m, 'num_features', 2048)
        for p in self.model.parameters(): p.requires_grad = False
    def forward(self, x): return self.model(x)

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

@torch.no_grad()
def extract_embeddings(backbone, df_subset):
    Xc, Xs, Ms, IDs = [], [], [], []
    bb = backbone.to(DEVICE).eval()

    def _load_img(path):
        try:
            im = Image.open(path).convert('RGB')
            return val_tf(im).unsqueeze(0).to(DEVICE)
        except:
            bgr = cv2.imread(str(path))
            if bgr is None:
                return torch.zeros(1,3,IMG_SIZE,IMG_SIZE, device=DEVICE)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb).convert('RGB')
            return val_tf(im).unsqueeze(0).to(DEVICE)

    for _, r in df_subset.iterrows():
        xcont = _load_img(r['contorno_path'])
        xsil = _load_img(r['sil_path']) if r['sil_path'] else xcont
        zc = bb(xcont).detach().cpu().numpy().astype(np.float32)
        zs = bb(xsil).detach().cpu().numpy().astype(np.float32)
        Mrow = np.array([r.get(c, np.nan) for c in MORPH_COLS], dtype=np.float32)
        Xc.append(zc); Xs.append(zs); Ms.append(Mrow[None,:]); IDs.append(r['sample_id'])

    Zc = np.concatenate(Xc, axis=0)
    Zs = np.concatenate(Xs, axis=0)
    M = np.concatenate(Ms, axis=0)
    col_means = np.nanmean(np.where(np.isnan(M), np.nan, M), axis=0)
    inds = np.where(np.isnan(M))
    M[inds] = np.take(col_means, inds[1])
    X = np.hstack([Zc, Zs, M])
    return IDs, X

backbone = BackboneTIMM(TIMM_MODEL)
print(f"\nBackbone: {TIMM_MODEL} | IMG_SIZE: {IMG_SIZE}")

IDs, X = extract_embeddings(backbone, df_new)
print(f"✅ Embeddings: shape={X.shape}")

# -----------------------------------
# CARGAR MODELOS Y PREDECIR
# -----------------------------------
json_models = sorted(EXPERIMENT_DIR.glob('xgboost_fold_*.json'))
json_models = [p for p in json_models if not p.stem.endswith('.config')]

pred_df = pd.DataFrame({'sample_id': IDs})
fold_preds = []

print(f"\n📥 Cargando {len(json_models)} modelos...")

for p in json_models:
    try:
        m = XGBRegressor()
        m.load_model(str(p))
        y = m.predict(X).astype(float)
        pred_df[p.stem] = y
        fold_preds.append(y)
        print(f"   ✓ {p.stem}")
    except Exception as e:
        print(f"   ✗ {p.stem}: {e}")

# Ensemble
stack = np.stack(fold_preds, axis=1)
pred_df['pred_mean'] = stack.mean(axis=1)
pred_df['pred_median'] = np.median(stack, axis=1)
pred_df['pred_std'] = stack.std(axis=1, ddof=1) if stack.shape[1] > 1 else 0.0
pred_df['pred_cv_%'] = (pred_df['pred_std'] / pred_df['pred_mean'].replace(0, np.nan)) * 100

# Reordenar columnas
col_order = ['sample_id', 'pred_mean', 'pred_median', 'pred_std', 'pred_cv_%'] + \
            [c for c in pred_df.columns if c.startswith('xgboost_fold_')]
pred_df = pred_df[[c for c in col_order if c in pred_df.columns]]

# -----------------------------------
# MOSTRAR RESULTADOS DETALLADOS
# -----------------------------------
print("\n" + "="*80)
print("📋 PREDICCIONES POR FOLD (TODAS LAS MUESTRAS)")
print("="*80)
print("\n" + "-"*120)
print(pred_df.to_string(index=False))
print("-"*120)

# -----------------------------------
# EXTRAER PESOS VERDADEROS
# -----------------------------------
print("\n" + "="*80)
print("🔍 EXTRAYENDO PESOS VERDADEROS DE NOMBRES DE ARCHIVO")
print("="*80)

true_weights = []
valid_for_metrics = True

for sample_id in pred_df['sample_id']:
    try:
        parts = sample_id.replace('_', '-').split('-')
        weight = None
        for part in parts:
            try:
                w = float(part)
                if 50 <= w <= 800:
                    weight = w
                    break
            except ValueError:
                continue
        
        if weight is not None:
            true_weights.append(weight)
        else:
            true_weights.append(np.nan)
            valid_for_metrics = False
    except:
        true_weights.append(np.nan)
        valid_for_metrics = False

pred_df['true_weight'] = true_weights

if valid_for_metrics and not pred_df['true_weight'].isna().all():
    print("✅ Pesos verdaderos extraídos exitosamente")
    
    # Calcular métricas del ensemble
    y_true = pred_df['true_weight'].values
    y_pred_ensemble = pred_df['pred_mean'].values
    
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred_ensemble)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred_ensemble[valid_mask]
    
    if len(y_true_valid) > 0:
        # RMSE
        rmse_ensemble = np.sqrt(np.mean((y_true_valid - y_pred_valid)**2))
        
        # MAE
        mae_ensemble = np.mean(np.abs(y_true_valid - y_pred_valid))
        
        # R²
        ss_res = np.sum((y_true_valid - y_pred_valid)**2)
        ss_tot = np.sum((y_true_valid - np.mean(y_true_valid))**2)
        r2_ensemble = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        
        # MAPE
        mape_ensemble = np.mean(np.abs((y_true_valid - y_pred_valid) / y_true_valid)) * 100
        
        # Max Error
        max_error_ensemble = np.max(np.abs(y_true_valid - y_pred_valid))
        
        print("\n" + "="*80)
        print("📊 MÉTRICAS DEL ENSEMBLE (PREDICCIÓN PROMEDIO)")
        print("="*80)
        print(f"   RMSE:      {rmse_ensemble:.3f} kg")
        print(f"   MAE:       {mae_ensemble:.3f} kg")
        print(f"   R²:        {r2_ensemble:.4f}")
        print(f"   MAPE:      {mape_ensemble:.2f}%")
        print(f"   Max Error: {max_error_ensemble:.3f} kg")
        
        # Calcular métricas por fold individual
        print("\n" + "="*80)
        print("📊 MÉTRICAS POR FOLD INDIVIDUAL")
        print("="*80)
        print(f"\n{'Fold':<20} {'RMSE (kg)':<15} {'MAE (kg)':<15} {'R²':<10} {'MAPE (%)':<10}")
        print("-"*70)
        
        fold_metrics = []
        fold_cols = [c for c in pred_df.columns if c.startswith('xgboost_fold_')]
        
        for fold_col in sorted(fold_cols):
            y_pred_fold = pred_df[fold_col].values[valid_mask]
            
            rmse_fold = np.sqrt(np.mean((y_true_valid - y_pred_fold)**2))
            mae_fold = np.mean(np.abs(y_true_valid - y_pred_fold))
            ss_res_fold = np.sum((y_true_valid - y_pred_fold)**2)
            r2_fold = 1 - (ss_res_fold / ss_tot) if ss_tot > 0 else np.nan
            mape_fold = np.mean(np.abs((y_true_valid - y_pred_fold) / y_true_valid)) * 100
            
            fold_metrics.append({
                'fold': fold_col,
                'rmse': rmse_fold,
                'mae': mae_fold,
                'r2': r2_fold,
                'mape': mape_fold
            })
            
            print(f"{fold_col:<20} {rmse_fold:<15.3f} {mae_fold:<15.3f} {r2_fold:<10.4f} {mape_fold:<10.2f}")
        
        # Resumen estadístico de folds
        fold_rmses = [m['rmse'] for m in fold_metrics]
        fold_maes = [m['mae'] for m in fold_metrics]
        fold_r2s = [m['r2'] for m in fold_metrics]
        fold_mapes = [m['mape'] for m in fold_metrics]
        
        print("-"*70)
        print(f"{'PROMEDIO':<20} {np.mean(fold_rmses):<15.3f} {np.mean(fold_maes):<15.3f} {np.mean(fold_r2s):<10.4f} {np.mean(fold_mapes):<10.2f}")
        print(f"{'DESV. STD':<20} {np.std(fold_rmses, ddof=1):<15.3f} {np.std(fold_maes, ddof=1):<15.3f} {np.std(fold_r2s, ddof=1):<10.4f} {np.std(fold_mapes, ddof=1):<10.2f}")
        print(f"{'MÍNIMO':<20} {np.min(fold_rmses):<15.3f} {np.min(fold_maes):<15.3f} {np.min(fold_r2s):<10.4f} {np.min(fold_mapes):<10.2f}")
        print(f"{'MÁXIMO':<20} {np.max(fold_rmses):<15.3f} {np.max(fold_maes):<15.3f} {np.max(fold_r2s):<10.4f} {np.max(fold_mapes):<10.2f}")
        
        # Comparación Ensemble vs Folds
        print("\n" + "="*80)
        print("📊 COMPARACIÓN: ENSEMBLE vs FOLDS INDIVIDUALES")
        print("="*80)
        print(f"\n{'Métrica':<15} {'Ensemble':<15} {'Folds (media)':<20} {'Mejora %':<15}")
        print("-"*65)
        
        mejora_rmse = ((np.mean(fold_rmses) - rmse_ensemble) / np.mean(fold_rmses)) * 100
        mejora_mae = ((np.mean(fold_maes) - mae_ensemble) / np.mean(fold_maes)) * 100
        mejora_r2 = ((r2_ensemble - np.mean(fold_r2s)) / abs(np.mean(fold_r2s))) * 100
        
        print(f"{'RMSE (kg)':<15} {rmse_ensemble:<15.3f} {np.mean(fold_rmses):<20.3f} {mejora_rmse:<15.2f}")
        print(f"{'MAE (kg)':<15} {mae_ensemble:<15.3f} {np.mean(fold_maes):<20.3f} {mejora_mae:<15.2f}")
        print(f"{'R²':<15} {r2_ensemble:<15.4f} {np.mean(fold_r2s):<20.4f} {mejora_r2:<15.2f}")
        print(f"{'MAPE (%)':<15} {mape_ensemble:<15.2f} {np.mean(fold_mapes):<20.2f}")
        
        # Tabla con errores individuales
        pred_df['error_kg'] = pred_df['true_weight'] - pred_df['pred_mean']
        pred_df['error_abs_kg'] = np.abs(pred_df['error_kg'])
        pred_df['error_%'] = (pred_df['error_abs_kg'] / pred_df['true_weight']) * 100
        
        # Guardar con métricas
        out_metrics_csv = PRED_DIR / 'predicciones_con_metricas.csv'
        pred_df.to_csv(out_metrics_csv, index=False)
        
        # Guardar JSON con resumen
        metrics_summary = {
            'ensemble': {
                'rmse_kg': float(rmse_ensemble),
                'mae_kg': float(mae_ensemble),
                'r2': float(r2_ensemble),
                'mape_%': float(mape_ensemble),
                'max_error_kg': float(max_error_ensemble)
            },
            'folds': {
                'rmse_mean': float(np.mean(fold_rmses)),
                'rmse_std': float(np.std(fold_rmses, ddof=1)),
                'rmse_min': float(np.min(fold_rmses)),
                'rmse_max': float(np.max(fold_rmses)),
                'mae_mean': float(np.mean(fold_maes)),
                'mae_std': float(np.std(fold_maes, ddof=1)),
                'r2_mean': float(np.mean(fold_r2s)),
                'r2_std': float(np.std(fold_r2s, ddof=1)),
                'mape_mean': float(np.mean(fold_mapes))
            },
            'improvement': {
                'rmse_%': float(mejora_rmse),
                'mae_%': float(mejora_mae),
                'r2_%': float(mejora_r2)
            }
        }
        
        out_json = PRED_DIR / 'metricas_inferencia.json'
        with open(out_json, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        print(f"\n📄 Tabla con métricas: {out_metrics_csv}")
        print(f"📄 Resumen JSON: {out_json}")
        
        # Análisis de peores predicciones
        print("\n" + "="*80)
        print("🔴 TOP 5 PEORES PREDICCIONES (Mayor error absoluto)")
        print("="*80)
        worst = pred_df.nlargest(5, 'error_abs_kg')[['sample_id', 'true_weight', 'pred_mean', 'error_kg', 'error_abs_kg', 'error_%', 'pred_cv_%']]
        print(worst.to_string(index=False))
        
        # Análisis de mejores predicciones
        print("\n" + "="*80)
        print("🟢 TOP 5 MEJORES PREDICCIONES (Menor error absoluto)")
        print("="*80)
        best = pred_df.nsmallest(5, 'error_abs_kg')[['sample_id', 'true_weight', 'pred_mean', 'error_kg', 'error_abs_kg', 'error_%', 'pred_cv_%']]
        print(best.to_string(index=False))
        
    else:
        print("⚠️ No se encontraron pesos válidos")
else:
    print("⚠️ No se pudieron extraer pesos verdaderos")
    print("   Formato esperado: ID-PESO.ext (ej: vaca001-450.5.jpg)")

# -----------------------------------
# RESUMEN ESTADÍSTICO GENERAL
# -----------------------------------
print("\n" + "="*80)
print("🎯 RESUMEN ESTADÍSTICO GENERAL (ENSEMBLE)")
print("="*80)
print(f"\n   Peso promedio:      {pred_df['pred_mean'].mean():.2f} ± {pred_df['pred_mean'].std():.2f} kg")
print(f"   Peso mediano:       {pred_df['pred_median'].mean():.2f} kg")
print(f"   Rango de pesos:     {pred_df['pred_mean'].min():.2f} - {pred_df['pred_mean'].max():.2f} kg")
print(f"   Desv. std promedio: {pred_df['pred_std'].mean():.2f} kg")
print(f"   CV% promedio:       {pred_df['pred_cv_%'].mean():.2f}%")
print(f"   CV% máximo:         {pred_df['pred_cv_%'].max():.2f}%")
print(f"   CV% mínimo:         {pred_df['pred_cv_%'].min():.2f}%")

# Análisis de incertidumbre
print("\n" + "="*80)
print("🔍 ANÁLISIS DE INCERTIDUMBRE")
print("="*80)

most_uncertain = pred_df.nlargest(3, 'pred_cv_%')[['sample_id', 'pred_mean', 'pred_std', 'pred_cv_%']]
least_uncertain = pred_df.nsmallest(3, 'pred_cv_%')[['sample_id', 'pred_mean', 'pred_std', 'pred_cv_%']]

print("\n🔴 Top 3 predicciones con MAYOR incertidumbre:")
print(most_uncertain.to_string(index=False))

print("\n🟢 Top 3 predicciones con MENOR incertidumbre:")
print(least_uncertain.to_string(index=False))

# Guardar archivo principal
out_csv = PRED_DIR / 'predicciones_baseline.csv'
pred_df.to_csv(out_csv, index=False)

print("\n" + "="*80)
print("✅ ANÁLISIS COMPLETADO")
print("="*80)
print(f"📄 Predicciones: {out_csv}")
if 'true_weight' in pred_df.columns and not pred_df['true_weight'].isna().all():
    print(f"📊 RMSE BASELINE: {rmse_ensemble:.3f} kg")
    print(f"📊 MAE BASELINE:  {mae_ensemble:.3f} kg")
    print(f"📊 R² BASELINE:   {r2_ensemble:.4f}")
else:
    print(f"📊 RMSE esperado: ~76 kg (baseline)")
print("="*80)
