from pathlib import Path
import torch
import warnings
import gc

# Suprimir warnings de XGBoost sobre pickle
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

# Forzar solo CPU para reducir uso de memoria
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

BASE_DIR = Path(__file__).resolve().parent.parent  
ARTEFACTOS = BASE_DIR / 'artefactos_modelo'

# Paths de modelos
CIRCLE_MODEL_PATH = ARTEFACTOS / 'last_actual.pt'
COW_MODEL_PATH    = ARTEFACTOS / 'yolov8x-seg.pt'

# Variables globales para lazy loading
_circle_model = None
_cow_model = None
_backbone = None
_xgb_models = None
_cow_id_cache = None

def get_device():
    """Obtener dispositivo (CPU solo)"""
    return torch.device('cpu')

def load_circle_model():
    """Lazy loading del modelo de detección de círculo"""
    global _circle_model
    if _circle_model is None:
        print("[MODELS] Cargando círculo YOLO...")
        import gc
        from ultralytics.models.yolo import YOLO
        
        # Configurar ultralytics para usar menos memoria
        import ultralytics
        ultralytics.checks.check_requirements = lambda x: None  # Skip checks
        
        _circle_model = YOLO(str(CIRCLE_MODEL_PATH))
        gc.collect()
        print("[MODELS] ✓ Círculo YOLO cargado")
    return _circle_model

def load_cow_model():
    """Lazy loading del modelo de detección de vaca"""
    global _cow_model
    if _cow_model is None:
        print("[MODELS] Cargando vaca YOLO...")
        import gc
        from ultralytics.models.yolo import YOLO
        _cow_model = YOLO(str(COW_MODEL_PATH))
        gc.collect()
        print("[MODELS] ✓ Vaca YOLO cargado")
    return _cow_model

def load_backbone():
    """Lazy loading del backbone WideResNet-50-2"""
    global _backbone
    if _backbone is None:
        print("[MODELS] Cargando backbone WideResNet-50-2...")
        from .backbone import BackboneTIMM
        device = torch.device('cpu')
        _backbone = BackboneTIMM(model_name='wide_resnet50_2', trainable=False)
        _backbone.to(device).eval()
        # Liberar memoria innecesaria
        gc.collect()
        print(f"[MODELS] ✓ Backbone WideResNet-50-2 cargado ({_backbone.out_dim} features)")
    return _backbone

def load_xgb_models():
    """Lazy loading del ensemble XGBoost"""
    global _xgb_models
    if _xgb_models is None:
        print("[MODELS] Cargando ensemble XGBoost (10 folds)...")
        from xgboost import XGBRegressor
        _xgb_models = []
        for fold_idx in range(1, 11):
            fold_path = ARTEFACTOS / f'xgboost_fold_{fold_idx}.json'
            if not fold_path.exists():
                print(f"[WARNING] No se encontró {fold_path.name}, se omitirá este fold")
                continue
            try:
                model = XGBRegressor()
                model.load_model(str(fold_path))
                _xgb_models.append((fold_idx, model))
                print(f"[MODELS] ✓ Fold {fold_idx} cargado")
            except Exception as e:
                print(f"[WARNING] Error al cargar fold {fold_idx}: {e}")
        
        if len(_xgb_models) == 0:
            raise RuntimeError("No se pudo cargar ningún fold de XGBoost. Verifica artefactos_modelo/")
        
        print(f"[MODELS] ✅ Ensemble XGBoost: {len(_xgb_models)}/10 folds cargados")
    return _xgb_models

# Aliases para compatibilidad (llamar directamente a las funciones)
circle_model = load_circle_model
cow_model = load_cow_model
backbone = load_backbone
xgb_models = load_xgb_models

# Helper para obtener id de 'cow'
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
