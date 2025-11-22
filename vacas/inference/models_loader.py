from pathlib import Path
from ultralytics.models.yolo import YOLO
import torch
import warnings

# Suprimir warnings de XGBoost sobre pickle
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

from .backbone import BackboneTIMM, DEVICE

BASE_DIR = Path(__file__).resolve().parent.parent  
ARTEFACTOS = BASE_DIR / 'artefactos_modelo'

# YOLOs
CIRCLE_MODEL_PATH = ARTEFACTOS / 'last_actual.pt'
COW_MODEL_PATH    = ARTEFACTOS / 'yolov8x-seg.pt'

circle_model = YOLO(str(CIRCLE_MODEL_PATH))
cow_model    = YOLO(str(COW_MODEL_PATH))

# Backbone (WideResNet-50-2) - Usar modelo pre-entrenado de timm
# Los XGBoost fueron entrenados con WideResNet-50-2 (2048 features)
# Total features: 2048 (contorno) + 2048 (silueta) + 9 (morph) = 4105
backbone = BackboneTIMM(model_name='wide_resnet50_2', trainable=False)
backbone.to(DEVICE).eval()
print(f"[MODELS] ✓ Backbone WideResNet-50-2 cargado ({backbone.out_dim} features)")

# XGBoost - Cargar los 10 folds completos (ensemble)
print(f"[MODELS] Cargando ensemble XGBoost (10 folds)...")
xgb_models = []
for fold_idx in range(1, 11):  # Folds 1-10
    fold_path = ARTEFACTOS / f'xgboost_fold_{fold_idx}.json'
    if not fold_path.exists():
        print(f"[WARNING] No se encontró {fold_path.name}, se omitirá este fold")
        continue
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model(str(fold_path))
        xgb_models.append((fold_idx, model))
        print(f"[MODELS] ✓ Fold {fold_idx} cargado")
    except Exception as e:
        print(f"[WARNING] Error al cargar fold {fold_idx}: {e}")

if len(xgb_models) == 0:
    raise RuntimeError("No se pudo cargar ningún fold de XGBoost. Verifica artefactos_modelo/")

print(f"[MODELS] ✅ Ensemble XGBoost: {len(xgb_models)}/10 folds cargados")

# helper sencillo para obtener id de 'cow'
_cow_id_cache = None
def get_cow_class_id_cached(model):
    global _cow_id_cache
    if _cow_id_cache is not None:
        return _cow_id_cache
    names = model.names
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v).lower() == 'cow':
                _cow_id_cache = int(k)
                break
    elif isinstance(names, (list, tuple)):
        for i, v in enumerate(names):
            if str(v).lower() == 'cow':
                _cow_id_cache = int(i)
                break
    return _cow_id_cache
