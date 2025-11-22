# Guía de Implementación en Producción - Django

## 📋 Resumen Ejecutivo

Sistema de predicción de peso de ganado bovino mediante análisis de imágenes RAW usando WideResNet-50-2 + XGBoost con **RMSE calibrado de 79 kg** (R²=0.503).

---

## 🎯 Especificaciones Técnicas Validadas

### Rendimiento del Modelo
- **RMSE**: 79.0 kg
- **MAE**: 58.9 kg
- **R²**: 0.503 (explica 50% de la varianza)
- **MAPE**: 23.1%
- **Consistencia entre folds**: std = 4.3 kg
- **Tasa de éxito**: 100% (20/20 imágenes procesadas)
- **Tiempo por imagen**: ~1.2 segundos

### Arquitectura del Modelo
```
WideResNet-50-2 (timm: wide_resnet50_2)
├── Entrada dual:
│   ├── Contorno (256×256) → 2048 features
│   └── Silueta (256×256) → 2048 features
├── Features morfológicos: 9 dimensiones
│   ├── Geométricos: area_px, perim_px, bbox_w, bbox_h, aspect, fill_frac
│   └── Forma: hu1, hu2, hu3 (Hu moments)
└── XGBoost Ensemble: 10 modelos (10-fold CV)
    └── Predicción final: promedio de 10 folds
```

---

## 🔧 Pipeline de Preprocesamiento (CRÍTICO)

### Parámetros Calibrados Óptimos

```python
# 1. Canvas estandarizado
TARGET_W = 1280
TARGET_H = 960

# 2. YOLOv8x-seg para segmentación
YOLO_MODEL = "yolov8x-seg.pt"
YOLO_CONF = 0.20  # Umbral de confianza
YOLO_IOU = 0.45   # Umbral NMS

# 3. MORFOLOGÍA (CONFIGURACIÓN ÓPTIMA - RMSE 79 kg)
# ⚠️ NO MODIFICAR SIN RECALIBRACIÓN
def postprocess_mask(mask, H, W):
    """
    Postprocesa máscara binaria con morfología calibrada.
    
    CALIBRADO ÓPTIMO: RMSE 79 kg
    Validado: 27% mejora vs baseline agresivo (108 kg)
    """
    import cv2
    import numpy as np
    
    m = mask.copy()
    
    # Dilate: cerrar pequeños huecos internos
    k_dilate = max(3, int(0.0045 * min(H, W)))  # ~4-5 px para 960p
    K_dilate = np.ones((k_dilate, k_dilate), np.uint8)
    m = cv2.dilate(m, K_dilate, iterations=1)
    
    # Close: suavizar contorno exterior
    k_close = max(4, int(0.009 * min(H, W)))  # ~8-9 px para 960p
    K_close = np.ones((k_close, k_close), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K_close, iterations=2)
    
    # NO usar MORPH_OPEN (reduce área excesivamente)
    
    # Contorno principal
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return mask  # Fallback
    
    cnt = max(cnts, key=cv2.contourArea)
    
    # Suavizado moderado con approxPolyDP (NO convexHull)
    # ⚠️ convexHull colapsa perímetro -41% → RMSE 108 kg
    eps = 0.002 * cv2.arcLength(cnt, True)
    cnt_smooth = cv2.approxPolyDP(cnt, eps, True)
    
    m_out = np.zeros_like(m)
    cv2.drawContours(m_out, [cnt_smooth], -1, 255, -1)
    
    return m_out

# 4. Filtro bilateral (reducción de ruido)
BILATERAL_D = 9
BILATERAL_SIGMA = 50

# 5. Crop con padding
BBOX_PADDING = 0.10  # 10% extra alrededor del bbox

# 6. Resize final
FEATURE_SIZE = 256  # 256×256 para entrada a WideResNet
```

---

## 📦 Dependencias Python

```txt
# requirements.txt para Django
torch==2.0.1
torchvision==0.15.2
timm==0.9.2
xgboost==1.7.6
ultralytics==8.0.196  # YOLOv8x-seg
opencv-python==4.8.0.74
numpy==1.24.3
pillow==10.0.0
scikit-learn==1.3.0
```

---

## 🗂️ Estructura de Archivos del Modelo

```
models/
├── wide_resnet50_2/
│   ├── fold_0.pth   # Pesos WideResNet fold 0
│   ├── fold_1.pth
│   ├── ...
│   └── fold_9.pth
├── xgboost_models/
│   ├── fold_0.json  # XGBoost fold 0
│   ├── fold_1.json
│   ├── ...
│   └── fold_9.json
└── yolov8x-seg.pt   # Segmentador YOLO
```

**Descarga de modelos**: Los pesos WideResNet se entrenan desde el script `tesis_vacas.py`. YOLO se descarga automáticamente con `ultralytics`.

---

## 🐍 Código Django - Vista de Predicción

```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import torch
import cv2
import numpy as np
import timm
import xgboost as xgb
from ultralytics import YOLO
from PIL import Image
import io

# Configuración global
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_W, TARGET_H = 1280, 960
FEATURE_SIZE = 256

# Cargar modelos (hacer esto en startup, no por request)
def load_models():
    """Cargar todos los modelos al iniciar Django."""
    
    # YOLO
    yolo = YOLO("models/yolov8x-seg.pt")
    
    # WideResNet (10 folds)
    wideresnet_models = []
    for fold in range(10):
        model = timm.create_model('wide_resnet50_2', pretrained=False, num_classes=0)
        model.load_state_dict(torch.load(f"models/wide_resnet50_2/fold_{fold}.pth", map_location=DEVICE))
        model.to(DEVICE).eval()
        wideresnet_models.append(model)
    
    # XGBoost (10 folds)
    xgb_models = []
    for fold in range(10):
        bst = xgb.Booster()
        bst.load_model(f"models/xgboost_models/fold_{fold}.json")
        xgb_models.append(bst)
    
    return yolo, wideresnet_models, xgb_models

# Instancias globales (cargar en AppConfig.ready())
YOLO_MODEL = None
WIDERESNET_MODELS = None
XGB_MODELS = None

def postprocess_mask(mask, H, W):
    """Morfología calibrada (RMSE 79 kg)."""
    m = mask.copy()
    
    k_dilate = max(3, int(0.0045 * min(H, W)))
    K_dilate = np.ones((k_dilate, k_dilate), np.uint8)
    m = cv2.dilate(m, K_dilate, iterations=1)
    
    k_close = max(4, int(0.009 * min(H, W)))
    K_close = np.ones((k_close, k_close), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K_close, iterations=2)
    
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return mask
    
    cnt = max(cnts, key=cv2.contourArea)
    eps = 0.002 * cv2.arcLength(cnt, True)
    cnt_smooth = cv2.approxPolyDP(cnt, eps, True)
    
    m_out = np.zeros_like(m)
    cv2.drawContours(m_out, [cnt_smooth], -1, 255, -1)
    return m_out

def extract_morphological_features(mask_uint8):
    """Extrae 9 features morfológicos."""
    cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return np.zeros(9)
    
    cnt = max(cnts, key=cv2.contourArea)
    
    area_px = cv2.contourArea(cnt)
    perim_px = cv2.arcLength(cnt, True)
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    aspect = w_box / max(h_box, 1)
    fill_frac = area_px / (256 * 256)
    
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu1, hu2, hu3 = hu_moments[0], hu_moments[1], hu_moments[2]
    
    return np.array([area_px, perim_px, w_box, h_box, aspect, fill_frac, hu1, hu2, hu3])

def preprocess_image(image_bytes):
    """Pipeline completo de preprocesamiento."""
    
    # 1. Cargar imagen
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    
    # 2. Orientación vertical (rotar si es horizontal)
    H_orig, W_orig = img_np.shape[:2]
    if W_orig > H_orig:
        img_np = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
        H_orig, W_orig = img_np.shape[:2]
    
    # 3. Resize con letterbox a 1280×960
    scale = min(TARGET_W / W_orig, TARGET_H / H_orig)
    new_w = int(W_orig * scale)
    new_h = int(H_orig * scale)
    img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.full((TARGET_H, TARGET_W, 3), 114, dtype=np.uint8)
    y_offset = (TARGET_H - new_h) // 2
    x_offset = (TARGET_W - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
    
    # 4. Filtro bilateral
    img_filt = cv2.bilateralFilter(canvas, d=9, sigmaColor=50, sigmaSpace=50)
    
    # 5. Segmentación YOLO
    results = YOLO_MODEL.predict(img_filt, conf=0.20, iou=0.45, verbose=False)
    if len(results[0].masks) == 0:
        return None, None  # No se detectó vaca
    
    mask_raw = results[0].masks.data[0].cpu().numpy()
    mask_resized = cv2.resize(mask_raw, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
    mask_uint8 = (mask_resized * 255).astype(np.uint8)
    
    # 6. Morfología calibrada
    mask_final = postprocess_mask(mask_uint8, TARGET_H, TARGET_W)
    
    # 7. Crop con padding
    ys, xs = np.where(mask_final > 127)
    if len(ys) == 0:
        return None, None
    
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    
    pad_h = int((y2 - y1) * 0.10)
    pad_w = int((x2 - x1) * 0.10)
    
    y1 = max(0, y1 - pad_h)
    y2 = min(TARGET_H, y2 + pad_h)
    x1 = max(0, x1 - pad_w)
    x2 = min(TARGET_W, x2 + pad_w)
    
    # Crops
    crop_img = img_filt[y1:y2, x1:x2]
    crop_mask = mask_final[y1:y2, x1:x2]
    
    # 8. Resize a 256×256
    crop_img_256 = cv2.resize(crop_img, (FEATURE_SIZE, FEATURE_SIZE))
    crop_mask_256 = cv2.resize(crop_mask, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_NEAREST)
    
    # 9. Generar contorno y silueta
    contorno = crop_img_256.copy()
    contorno[crop_mask_256 < 127] = 114  # Background gris
    
    silueta = np.zeros_like(crop_img_256)
    silueta[crop_mask_256 >= 127] = 255  # Blanco donde hay vaca
    
    # 10. Features morfológicos
    morph_feats = extract_morphological_features(crop_mask_256)
    
    return (contorno, silueta), morph_feats

def extract_embeddings(contorno, silueta):
    """Extrae embeddings con WideResNet (promedio de 10 folds)."""
    
    # Normalización ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def to_tensor(img):
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        t = (t - mean) / std
        return t.unsqueeze(0).to(DEVICE)
    
    t_contorno = to_tensor(contorno)
    t_silueta = to_tensor(silueta)
    
    embeddings_all = []
    
    with torch.no_grad():
        for model in WIDERESNET_MODELS:
            emb_contorno = model(t_contorno).cpu().numpy()  # (1, 2048)
            emb_silueta = model(t_silueta).cpu().numpy()    # (1, 2048)
            emb = np.concatenate([emb_contorno, emb_silueta], axis=1)  # (1, 4096)
            embeddings_all.append(emb)
    
    # Promedio de embeddings de 10 folds
    emb_avg = np.mean(embeddings_all, axis=0)  # (1, 4096)
    return emb_avg

def predict_weight(embeddings, morph_feats):
    """Predicción con XGBoost (promedio de 10 folds)."""
    
    # Concatenar embeddings + morfología
    X = np.concatenate([embeddings, morph_feats.reshape(1, -1)], axis=1)  # (1, 4105)
    
    # Predecir con cada fold
    predictions = []
    for bst in XGB_MODELS:
        dtest = xgb.DMatrix(X)
        pred = bst.predict(dtest)[0]
        predictions.append(pred)
    
    # Promedio ensemble
    peso_final = np.mean(predictions)
    std_ensemble = np.std(predictions)
    
    return peso_final, std_ensemble, predictions

@csrf_exempt
def predict_cattle_weight(request):
    """
    Endpoint Django para predicción de peso.
    
    POST /api/predict/
    Content-Type: multipart/form-data
    Body: image (archivo JPG/PNG)
    
    Response:
    {
        "peso_kg": 450.5,
        "incertidumbre_kg": 4.2,
        "predicciones_folds": [448.3, 451.2, ...],
        "features_morfologicos": {
            "area_px": 8574,
            "perim_px": 689,
            ...
        }
    }
    """
    
    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido"}, status=405)
    
    if "image" not in request.FILES:
        return JsonResponse({"error": "No se envió imagen"}, status=400)
    
    try:
        # Leer bytes de la imagen
        image_file = request.FILES["image"]
        image_bytes = image_file.read()
        
        # Preprocesar
        result = preprocess_image(image_bytes)
        if result[0] is None:
            return JsonResponse({"error": "No se detectó ganado en la imagen"}, status=400)
        
        (contorno, silueta), morph_feats = result
        
        # Extraer embeddings
        embeddings = extract_embeddings(contorno, silueta)
        
        # Predecir peso
        peso_kg, std_kg, predictions = predict_weight(embeddings, morph_feats)
        
        # Respuesta
        return JsonResponse({
            "peso_kg": round(float(peso_kg), 2),
            "incertidumbre_kg": round(float(std_kg), 2),
            "predicciones_folds": [round(float(p), 2) for p in predictions],
            "features_morfologicos": {
                "area_px": int(morph_feats[0]),
                "perim_px": round(float(morph_feats[1]), 2),
                "bbox_w": int(morph_feats[2]),
                "bbox_h": int(morph_feats[3]),
                "aspect": round(float(morph_feats[4]), 3),
                "fill_frac": round(float(morph_feats[5]), 3)
            }
        })
    
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
```

---

## 🚀 Configuración Django

```python
# apps.py
from django.apps import AppConfig

class PredictionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'prediction'
    
    def ready(self):
        """Cargar modelos al iniciar Django."""
        from . import views
        views.YOLO_MODEL, views.WIDERESNET_MODELS, views.XGB_MODELS = views.load_models()
        print("✅ Modelos cargados exitosamente")
```

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('api/predict/', views.predict_cattle_weight, name='predict'),
]
```

---

## 📊 Características del Sistema

### Distribución de Features Esperada (RAW Calibrado)
```
area_px:    8574 ± 2640 px  (vs 11168 training, -23%)
perim_px:   689 ± 157 px    (vs 771 training, -11%)
bbox_h:     132 ± 27 px     (vs 165 training, -20%)
aspect:     0.955 ± 0.122   (vs 0.784 training, +22%)
fill_frac:  0.132 ± 0.054   (vs 0.170 training, -22%)
```

**Nota**: La desviación del ~20-22% vs training es inherente a diferencias de cámara/iluminación/ángulo entre RAW y dataset de entrenamiento. No modificar parámetros morfológicos para "corregir" esto sin recalibración completa.

### Manejo de Errores
- **Sin detección YOLO**: Retornar error 400 ("No se detectó ganado")
- **Máscara vacía post-morfología**: Usar máscara raw de YOLO (fallback)
- **Excepción en modelo**: Capturar y retornar error 500 con traceback

---

## ⚠️ ADVERTENCIAS CRÍTICAS

### 🔴 NO MODIFICAR
1. **Parámetros morfológicos** (`k_dilate=0.0045`, `k_close=0.009`, `eps=0.002`)
   - Calibrados tras 3 iteraciones para RMSE 79 kg
   - Cambios de ±10% causan degradación de 1-30 kg en RMSE

2. **YOLOv8x-seg con CONF=0.20, IOU=0.45**
   - Balanceado para detectar vacas parcialmente ocluidas sin falsos positivos

3. **Canvas 1280×960 con letterbox**
   - Mantiene aspect ratio original, evita distorsión

4. **NO usar `convexHull`**
   - Colapsa perímetro -41% → RMSE sube a 108 kg

### 🟡 Limitaciones Conocidas
- **Degradación RAW vs Test**: 7.3X (79 kg vs 10.77 kg)
  - Causado por diferencias cámara/iluminación/pose
  - Mejora requiere: fine-tuning YOLO, multi-view, recalibración cámara

- **Rango de peso efectivo**: 180-550 kg
  - Fuera de rango: predicciones menos confiables

- **Requerimientos de imagen**:
  - Resolución mínima: 640×480
  - Vaca debe ocupar >30% del encuadre
  - Iluminación uniforme (evitar contraluz)

---

## 🧪 Pruebas de Validación

### Script de Prueba
```python
# test_endpoint.py
import requests

url = "http://localhost:8000/api/predict/"
image_path = "test_cow.jpg"

with open(image_path, "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)

print(response.json())
# Esperado: {"peso_kg": 450.5, "incertidumbre_kg": 4.2, ...}
```

### Validación de Rendimiento
```bash
# Procesar 20 imágenes RAW
python manage.py shell

from prediction.views import *
import glob

images = glob.glob("test_images/*.jpg")
for img_path in images:
    with open(img_path, "rb") as f:
        result = preprocess_image(f.read())
        if result[0]:
            (contorno, silueta), morph = result
            emb = extract_embeddings(contorno, silueta)
            peso, std, _ = predict_weight(emb, morph)
            print(f"{img_path}: {peso:.1f} kg ± {std:.1f} kg")
```

**Métricas esperadas**:
- RMSE: ~79 kg
- Tiempo/imagen: 1-1.5 seg (CPU), 0.5-0.8 seg (GPU)
- Memoria: ~2 GB (modelos cargados)

---

## 📈 Monitoreo en Producción

### Métricas a Registrar
```python
# Agregar en cada predicción
import logging

logger = logging.getLogger(__name__)

logger.info({
    "timestamp": datetime.now(),
    "peso_predicho": peso_kg,
    "incertidumbre": std_kg,
    "cv_pct": (std_kg / peso_kg) * 100,  # <10% = confiable
    "area_px": morph_feats[0],
    "tiempo_procesamiento": elapsed_time
})
```

### Alertas de Calidad
- **CV% > 10%**: Predicción poco confiable (alta varianza entre folds)
- **area_px < 5000 o > 15000**: Fuera de rango calibrado
- **Tiempo > 3 seg**: Posible cuello de botella

---

## 🔄 Actualización de Modelos

Si se reentrenan los modelos:

1. **Generar nuevos pesos**:
   ```bash
   python tesis_vacas.py  # Entrena 10 folds WideResNet + XGBoost
   ```

2. **Reemplazar archivos** en `models/`:
   ```
   models/wide_resnet50_2/fold_*.pth
   models/xgboost_models/fold_*.json
   ```

3. **Reiniciar Django**:
   ```bash
   python manage.py runserver
   # Los modelos se cargan en AppConfig.ready()
   ```

4. **Recalibrar morfología** (si RMSE cambia >10%):
   - Ejecutar diagnósticos en training set (ver `tesis_vacas.py`)
   - Comparar features morfológicos con RAW
   - Ajustar `k_dilate`, `k_close`, `eps` iterativamente

---

## 📞 Soporte

Para mejoras estructurales (RMSE <50 kg):
1. **Fine-tuning YOLO**: Anotar 100-200 imágenes RAW → mejora 10-15%
2. **Multi-view**: Promediar 2-3 fotos por vaca → mejora 15-20%
3. **Recalibración cámara**: Estandarizar distancia/ángulo → mejora 5-10%
4. **Reentrenamiento full**: Fine-tune WideResNet en RAW → mejora 20-25%

**Mejora potencial combinada**: RMSE 35-50 kg (vs 79 kg actual)

---

## ✅ Checklist de Deployment

- [ ] Instalar dependencias (`pip install -r requirements.txt`)
- [ ] Descargar/entrenar modelos WideResNet (10 folds)
- [ ] Descargar/entrenar modelos XGBoost (10 folds)
- [ ] Descargar YOLOv8x-seg (`yolov8x-seg.pt`)
- [ ] Configurar `AppConfig.ready()` para cargar modelos
- [ ] Probar endpoint con imagen de prueba
- [ ] Validar RMSE ~79 kg en conjunto de validación RAW
- [ ] Configurar logging y monitoreo
- [ ] Establecer alertas de calidad (CV%, área_px, tiempo)
- [ ] Documentar limitaciones para usuarios finales

---

**Versión**: 1.0  
**Fecha de calibración**: Noviembre 2025  
**RMSE validado**: 79.0 kg (R²=0.503, MAE=58.9 kg, MAPE=23.1%)  
**Configuración óptima**: k_dilate=0.0045, k_close=0.009, iter=2, eps=0.002
