import numpy as np
import cv2
from typing import Optional, Tuple
from pathlib import Path

# ============================================================
# Constantes compartidas con el entrenamiento
# ============================================================
# IMPORTANTE: Estos valores DEBEN coincidir EXACTAMENTE con los usados en Google Colab
# ⚠️ CRÍTICO: Canvas debe ser (1280, 960) como en el entrenamiento
FIXED_CANVAS_WH: Tuple[int, int] = (1280, 960)  # (W, H) - IGUAL QUE COLAB
SIL_SIZE: Tuple[int, int] = (256, 256)           # tamaño de la silueta final
DRAW_CONTOUR_THICKNESS: int = 2                  # grosor del contorno verde

# Parámetros YOLO (Colab sección 3)
IMGSZ: int = 960   # Tamaño para inferencia YOLO
CONF: float = 0.30  # Confianza mínima
IOU: float = 0.5    # IoU threshold para NMS


# ============================================================
# Utilidades geométricas básicas
# ============================================================
def rotate_if_vertical(bgr: np.ndarray) -> np.ndarray:
    """
    Gira 90° CCW si la imagen está vertical (alto > ancho).
    Conserva exactamente la lógica usada en el notebook.
    """
    return (
        cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if bgr.shape[0] > bgr.shape[1] else bgr
    )


def resize_pad_fixed(img_bgr: np.ndarray,
                     target_wh: Tuple[int, int]) -> Tuple[np.ndarray, float, int, int]:
    """
    Letterbox a tamaño fijo target_wh=(W,H) conservando aspecto.
    Retorna: img_lb, scale, dx, dy  (por si quisieras mapear coords).
    """
    H, W = img_bgr.shape[:2]
    tw, th = target_wh
    r = min(tw / W, th / H)
    nw, nh = int(round(W * r)), int(round(H * r))

    img_rs = cv2.resize(
        img_bgr,
        (nw, nh),
        interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_CUBIC
    )
    canvas = np.full((th, tw, 3), 255, dtype=np.uint8)  # fondo blanco
    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = img_rs
    return canvas, r, dx, dy


# ============================================================
# Postpro de máscaras
# ============================================================
def keep_largest_component(bin_mask: np.ndarray) -> np.ndarray:
    """
    Conserva solo el componente conexo más grande de una máscara binaria.
    Devuelve máscara uint8 en {0,255}.
    """
    m = (bin_mask > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(m)
    if num <= 1:
        return m.astype(np.uint8) * 255

    best_label, best_area = 0, -1
    for lb in range(1, num):
        area = int((labels == lb).sum())
        if area > best_area:
            best_area, best_label = area, lb

    out = (labels == best_label).astype(np.uint8) * 255
    return out


def postprocess_mask(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Limpieza morfológica + relleno de huecos + suavizado de contorno.
    Recuperado tal cual de tu notebook.
    """
    m = (mask > 0).astype(np.uint8) * 255

    k = max(3, int(0.008 * min(H, W)))
    K = np.ones((k, k), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  K, iterations=1)

    # Relleno de huecos vía flood fill desde esquina
    mm = m.copy()
    ffmask = np.zeros((H + 2, W + 2), np.uint8)
    cv2.floodFill(mm, ffmask, (0, 0), 255)
    holes = cv2.bitwise_not(mm)
    m = cv2.bitwise_or(m, holes)

    # Quedarse solo con el componente mayor
    m = keep_largest_component(m)

    # Suavizar contorno
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        big = max(cnts, key=cv2.contourArea)
        eps = 0.003 * cv2.arcLength(big, True)
        smooth = cv2.approxPolyDP(big, eps, True)
        m[:] = 0
        cv2.drawContours(m, [smooth], -1, 255, thickness=cv2.FILLED)

    return m


def pick_best_mask(
    r,
    H: int,
    W: int,
    cc: Optional[Tuple[int, int, int]] = None,
    min_area_frac: float = 0.008,
    min_overlap_frac: float = 0.25,       # exigir solape con disco del sticker
    require_center_inside: bool = True,
    center_bonus_weight: float = 0.15
) -> Optional[np.ndarray]:
    """
    Selecciona la mejor máscara de YOLO (segmentación) siguiendo tu criterio:
    - área mínima
    - solape con disco del sticker
    - bonus por centro dentro y proximidad al centro de la imagen
    """
    if r is None or r.masks is None or len(r.masks.data) == 0:
        return None

    raw = r.masks.data.cpu().numpy().astype(np.uint8)
    masks = [cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) for m in raw]

    if getattr(r, "boxes", None) is not None and len(r.boxes) == len(masks):
        confs = r.boxes.conf.cpu().numpy().tolist()
    else:
        confs = [0.0] * len(masks)

    area_min = int(min_area_frac * H * W)
    keep = [(i, m) for i, m in enumerate(masks) if (m > 0).sum() >= area_min]
    if not keep:
        return None

    idxs, masks = zip(*keep)
    confs = [confs[i] for i in idxs]

    # Si tenemos círculo de referencia, usar solape
    if cc is not None:
        cx, cy, rr = cc
        disk = np.zeros((H, W), np.uint8)
        rcheck = max(6, int(rr * 0.6))
        cv2.circle(disk, (cx, cy), rcheck, 255, thickness=cv2.FILLED)
        disk_area = float((disk > 0).sum())

        candidates = []
        for i, m in enumerate(masks):
            inter = float(((m > 0) & (disk > 0)).sum())
            overlap = inter / max(1.0, disk_area)
            area = float((m > 0).sum())
            conf = float(confs[i])

            if require_center_inside:
                dil = cv2.dilate((m > 0).astype(np.uint8),
                                 np.ones((5, 5), np.uint8),
                                 iterations=1)
                inside = (
                    cy >= 0 and cy < H and cx >= 0 and cx < W and dil[cy, cx] > 0
                )
            else:
                inside = True

            candidates.append((i, overlap, area, conf, inside))

        valid = [
            (i, ov, ar, cf) for (i, ov, ar, cf, inside) in candidates
            if inside and ov >= min_overlap_frac
        ]
        if valid:
            i = max(valid, key=lambda t: (t[1], t[2], t[3]))[0]
            return masks[i].astype(np.uint8) * 255

        if candidates:
            i = max(candidates, key=lambda t: (t[1], t[2], t[3]))[0]
            return masks[i].astype(np.uint8) * 255

    # Si no hay círculo (o nada válido con círculo), criterio por área + centro
    best_i, best_score = None, -1e18
    for i, m in enumerate(masks):
        area = float((m > 0).sum())
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        cxm, cym = xs.mean(), ys.mean()
        center_bonus = -((cxm / W - 0.5) ** 2 + (cym / H - 0.5) ** 2)
        score = area + center_bonus_weight * area * center_bonus + 2000.0 * confs[i]
        if score > best_score:
            best_score, best_i = score, i

    return masks[best_i].astype(np.uint8) * 255 if best_i is not None else None


# ============================================================
# Detección de círculo en canvas fijo
# ============================================================
def detect_circle_in_canvas(
    bgr_fixed: np.ndarray,
    circle_model=None,
    debug_dir: Optional[Path] = None
) -> Optional[Tuple[int, int, int]]:
    """
    Detecta (cx, cy, r) del sticker usando EXCLUSIVAMENTE el modelo YOLO last_actual.pt
    
    NO usa ningún fallback (Hough, color, etc.). Si YOLO no detecta, retorna None.
    Esto mantiene la coherencia con el pipeline de entrenamiento en Colab.
    """
    if circle_model is None:
        print(f"[YOLO-CIRCLE] ❌ No se proporcionó modelo YOLO para detectar círculo")
        return None

    H, W = bgr_fixed.shape[:2]
    
    try:
        # Predicción YOLO (EXACTA configuración del Colab)
        # Colab usa: imgsz=min(max(W,H), 960) con canvas 1280x960 -> imgsz=1280
        # IMPORTANTE: No forzar imgsz fijo, dejar que se ajuste automáticamente
        results = circle_model.predict(
            bgr_fixed[:, :, ::-1],  # BGR -> RGB (Colab lo hace así)
            imgsz=min(max(W, H), 960),  # IGUAL que Colab: dinámico según imagen
            conf=0.20,  # Colab usa 0.20, mantener coherencia
            verbose=False,
            retina_masks=True
        )
        
        if not results or len(results) == 0:
            print(f"[YOLO-CIRCLE] ⚠️ YOLO retornó lista vacía")
            return None
            
        r = results[0]
        
        # Debug: mostrar qué detectó YOLO
        num_detections = 0
        if r.boxes is not None:
            num_detections = len(r.boxes)
        if r.masks is not None:
            num_detections = max(num_detections, len(r.masks.data))
        
        if num_detections > 0:
            print(f"[YOLO-CIRCLE] Detectó {num_detections} objeto(s), analizando...")
            if r.boxes is not None and len(r.boxes) > 0:
                confs = r.boxes.conf.cpu().numpy()
                print(f"[YOLO-CIRCLE] Confidencias: {[f'{c:.3f}' for c in confs]}")
        else:
            print(f"[YOLO-CIRCLE] Sin detecciones (todas con conf < 0.10)")

        # Caso 1: Segmentación con máscaras (preferido)
        if r.masks is not None and len(r.masks.data) > 0:
            # Obtener la primera máscara (más confiable)
            mask_points = r.masks.xy[0]  # Nx2 array
            
            if len(mask_points) < 3:
                print(f"[YOLO-CIRCLE] ⚠️ Máscara con muy pocos puntos: {len(mask_points)}")
                return None
            
            # Convertir a float32 explícitamente (compatibilidad OpenCV)
            pts = np.asarray(mask_points, dtype=np.float32).reshape(-1, 1, 2)
            
            # minEnclosingCircle requiere formato específico
            (cx, cy), rr = cv2.minEnclosingCircle(pts)
            
            # Validar que el círculo sea razonable
            if rr < 5 or rr > min(W, H) / 2:
                print(f"[YOLO-CIRCLE] ⚠️ Radio inválido: {rr} (debe estar entre 5 y {min(W,H)/2})")
                return None
            
            circ = (int(round(cx)), int(round(cy)), int(round(rr)))
            print(f"[YOLO-CIRCLE] ✅ Detectado via máscaras: centro=({circ[0]},{circ[1]}) r={circ[2]}")
            return circ
        
        # Caso 2: Solo bounding boxes (alternativo)
        elif r.boxes is not None and len(r.boxes) > 0:
            b = r.boxes[0]
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            rr = 0.5 * max((x2 - x1), (y2 - y1))
            
            # Validar bbox
            if rr < 5 or rr > min(W, H) / 2:
                print(f"[YOLO-CIRCLE] ⚠️ Radio de bbox inválido: {rr}")
                return None
            
            circ = (int(round(cx)), int(round(cy)), int(round(rr)))
            print(f"[YOLO-CIRCLE] ✅ Detectado via bbox: centro=({circ[0]},{circ[1]}) r={circ[2]}")
            return circ
        
        else:
            print(f"[YOLO-CIRCLE] ⚠️ YOLO no detectó ningún círculo (conf < 0.10)")
            return None
            
    except Exception as e:
        import traceback
        print(f"[YOLO-CIRCLE] ❌ Error en predicción: {e}")
        print(traceback.format_exc())
        return None


# ============================================================
# Normalización por círculo de referencia (canvas fijo)
# ============================================================
def normalize_by_circle(
    bgr: np.ndarray,
    circle_model=None,
    fixed_canvas_wh: Tuple[int, int] = FIXED_CANVAS_WH,
    debug_dir: Optional[Path] = None
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int]]]:
    """
    1) Rotar si la imagen está vertical.
    2) Letterbox a fixed_canvas_wh (todas quedan del mismo tamaño).
    3) Detectar círculo en ese canvas y devolver coords en el canvas.

    Devuelve:
        - imagen_normalizada (canvas fijo)
        - circ = (cx, cy, r) en coords del canvas, o None si no se detecta
    """
    img = rotate_if_vertical(bgr)
    fixed, _, _, _ = resize_pad_fixed(img, fixed_canvas_wh)
    circ = detect_circle_in_canvas(fixed, circle_model=circle_model, debug_dir=debug_dir)
    return fixed, circ


# ============================================================
# Segmentación de vaca + generación de contorno y silueta
# ============================================================
def segmentar_vaca_y_generar_contorno_y_silueta(
    norm_bgr: np.ndarray,
    cow_model,
    circle_model=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A partir de una imagen normalizada (canvas fijo) y un modelo YOLO de vaca,
    devuelve:
        - contorno_bgr: imagen con contorno verde de la vaca
        - silueta_bgr : imagen blanca con la silueta roja (0,0,255)

    Sigue la misma lógica que tu bloque de segmentación en Colab.
    """
    H, W = norm_bgr.shape[:2]

    # 1) Detectar círculo de referencia en el canvas fijo (opcional)
    circ = detect_circle_in_canvas(norm_bgr, circle_model=circle_model) if circle_model is not None else None

    # 2) Predicción YOLO de vaca (config EXACTA del Colab)
    rgb = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2RGB)
    r = cow_model.predict(
        rgb,
        imgsz=IMGSZ,  # Colab usa IMGSZ = 960
        conf=CONF,    # Colab usa CONF = 0.30
        iou=IOU,      # Colab usa IOU = 0.5
        verbose=False,
        retina_masks=True
    )[0]

    best = pick_best_mask(r, H, W, cc=circ)
    if best is None:
        raise RuntimeError("No se pudo obtener una máscara válida de la vaca.")

    best = postprocess_mask(best, H, W)

    # 3) Contorno verde sobre la imagen normalizada
    contorno = norm_bgr.copy()
    contours, _ = cv2.findContours(best, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smooth = []
    for c in contours:
        eps = 0.003 * cv2.arcLength(c, True)
        smooth.append(cv2.approxPolyDP(c, eps, True))
    if smooth:
        cv2.drawContours(contorno, smooth, -1, (0, 255, 0), thickness=DRAW_CONTOUR_THICKNESS)

    # 4) Silueta: fondo blanco, cuerpo rojo (como en tu pipeline)
    out_img = np.full((H, W, 3), 255, np.uint8)
    out_img[best > 0] = (0, 0, 255)
    silueta = cv2.resize(out_img, SIL_SIZE, interpolation=cv2.INTER_NEAREST)

    return contorno, silueta


# ============================================================
# Features morfológicos desde una silueta (imagen)
# ============================================================
def morph_from_silhouette_img(img_bgr: np.ndarray) -> dict:
    """
    Versión adaptada de morph_from_silhouette de tu notebook, pero recibiendo
    directamente la imagen BGR de la silueta en vez de un path.
    """
    def _nanrow():
        return dict(
            area_px=np.nan,
            perim_px=np.nan,
            bbox_w=np.nan,
            bbox_h=np.nan,
            aspect=np.nan,
            hu1=np.nan,
            hu2=np.nan,
            hu3=np.nan,
            fill_frac=np.nan
        )

    if img_bgr is None:
        return _nanrow()

    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Rango rojo (dos bandas en HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 50, 50]),   np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    mask = (mask > 0).astype(np.uint8)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return dict(
            area_px=0.0, perim_px=0.0, bbox_w=0.0, bbox_h=0.0,
            aspect=0.0, hu1=0.0, hu2=0.0, hu3=0.0, fill_frac=0.0
        )

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perim = float(cv2.arcLength(c, True))
    x, y, w, h = cv2.boundingRect(c)
    aspect = (w / h) if h > 0 else 0.0

    hu = cv2.HuMoments(cv2.moments(c)).flatten()[:3]
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    fill_frac = area / float(H * W)

    return dict(
        area_px=area,
        perim_px=perim,
        bbox_w=float(w),
        bbox_h=float(h),
        aspect=float(aspect),
        hu1=float(hu[0]),
        hu2=float(hu[1]),
        hu3=float(hu[2]),
        fill_frac=float(fill_frac)
    )
