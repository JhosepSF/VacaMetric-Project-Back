import numpy as np
import cv2
from PIL import Image
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from .backbone import val_tf, DEVICE, MORPH_COLS
from .preprocess import (
    normalize_by_circle,
    segmentar_vaca_y_generar_contorno_y_silueta,
    morph_from_silhouette_img
)
from . import models_loader

# Carpeta para guardar imágenes de debug
DEBUG_DIR = Path(__file__).parent.parent / 'debug_images'
DEBUG_DIR.mkdir(exist_ok=True)

def estimate_weight_from_bytes(file_obj, save_debug=False) -> Dict[str, Any]:
    """
    Estima el peso de una vaca desde bytes de imagen.
    
    Returns:
        Dict con:
        - peso_estimado_kg: float
        - confianza: str (alta/media/baja según desviación estándar)
        - circulo_detectado: bool
        - circulo_info: dict con detalles del círculo
        - advertencias: list de strings con advertencias
        - detalles: dict con información adicional del ensemble
    """
    # Leer bytes a BGR (OpenCV)
    data = file_obj.read()
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("No se pudo decodificar la imagen enviada.")

    # Extraer información del nombre del archivo (formato: idvaca-peso.jpg)
    filename = getattr(file_obj, 'name', 'unknown.jpg')
    file_stem = Path(filename).stem  # Sin extensión
    
    # Intentar extraer ID y peso real del nombre
    id_vaca = "unknown"
    peso_real = None
    peso_str = "NA"
    try:
        # Formato: idvaca-peso.jpg (ej: 1059-269.jpg)
        if '-' in file_stem:
            parts = file_stem.split('-')
            if len(parts) >= 2:
                id_vaca = parts[0]
                peso_str = parts[1]
                peso_real = float(parts[1])
        # Fallback: formato con guión bajo idvaca_peso.jpg
        elif '_' in file_stem:
            parts = file_stem.split('_')
            if len(parts) >= 2:
                id_vaca = parts[0]
                peso_str = parts[1]
                peso_real = float(parts[1])
    except (ValueError, IndexError):
        pass
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Nombre base: idvaca-pesoreal_timestamp (para identificación única)
    base_name = f"{id_vaca}-{peso_str}_{timestamp}" if id_vaca != "unknown" else timestamp
    
    # Información de debug (sin guardar imagen)
    if save_debug:
        print(f"[DEBUG] Imagen original: {filename}")
        if peso_real is not None:
            print(f"[DEBUG] ID Vaca: {id_vaca} | Peso Real: {peso_real:.2f} kg")
        print(f"[DEBUG] Procesando: {base_name}")

    # 1) Normalizar por círculo (canvas fijo)
    print(f"\n[PIPELINE] Iniciando detección del círculo de referencia...")
    norm_img, circ = normalize_by_circle(bgr, circle_model=models_loader.load_circle_model(), debug_dir=DEBUG_DIR if save_debug else None)
    print(f"[PIPELINE] Normalización completa.\n")
    
    # Información de círculo (sin guardar imagen)
    if save_debug:
        if circ is not None:
            cx, cy, r = circ
            print(f"[DEBUG] Círculo detectado: centro=({cx}, {cy}), radio={r}")
        else:
            print(f"[DEBUG] ⚠️ NO se detectó el círculo de referencia")

    # 2) Segmentar vaca y obtener contorno + silueta
    contorno, silueta = segmentar_vaca_y_generar_contorno_y_silueta(
        norm_img, models_loader.load_cow_model(), circle_model=models_loader.load_circle_model()
    )
    
    # Información de segmentación (sin guardar imágenes)
    if save_debug:
        print(f"[DEBUG] Segmentación completada: contorno y silueta generados")

    # 3) Features morfológicos desde silueta
    morph_dict = morph_from_silhouette_img(silueta)
    morph_vec = np.array([morph_dict[c] for c in MORPH_COLS], dtype=np.float32)

    # 4) Contorno y silueta -> PIL -> tensor
    cont_pil = Image.fromarray(cv2.cvtColor(contorno, cv2.COLOR_BGR2RGB))
    sil_pil  = Image.fromarray(cv2.cvtColor(silueta,  cv2.COLOR_BGR2RGB))

    # val_tf aplica transforms que convierten PIL Image a Tensor
    cont_tensor: torch.Tensor = val_tf(cont_pil)  # type: ignore
    sil_tensor: torch.Tensor = val_tf(sil_pil)  # type: ignore
    
    x_cont = cont_tensor.unsqueeze(0).to(DEVICE)  # (1,3,224,224)
    x_sil  = sil_tensor.unsqueeze(0).to(DEVICE)

    # 5) Embeddings WideResNet-50-2 (2048 features por imagen)
    backbone_model = models_loader.load_backbone()
    backbone_model.eval()
    with torch.no_grad():
        zc = backbone_model(x_cont).cpu().numpy()   # (1,2048)
        zs = backbone_model(x_sil).cpu().numpy()    # (1,2048)

    # 6) Vector final de features = [Zc, Zs, morph] = 1280+1280+9 = 2569
    feat = np.concatenate([zc, zs, morph_vec[None, :]], axis=1)  # (1, 2569)

    # 7) Predicción con TODOS los folds (ensemble)
    xgb_ensemble = models_loader.load_xgb_models()
    print(f"\n[ENSEMBLE] Prediciendo con {len(xgb_ensemble)} folds...")
    predicciones = []
    folds_info = []
    
    for fold_idx, model in xgb_ensemble:
        peso_fold = float(model.predict(feat)[0])
        predicciones.append(peso_fold)
        folds_info.append({'fold': fold_idx, 'peso_kg': peso_fold})
        print(f"  Fold {fold_idx}: {peso_fold:.2f} kg")
    
    # Promedio de todas las predicciones
    peso_promedio = float(np.mean(predicciones))
    peso_min = float(np.min(predicciones))
    peso_max = float(np.max(predicciones))
    peso_std = float(np.std(predicciones))
    
    # Calcular error si tenemos peso real
    error_absoluto = None
    error_porcentual = None
    if peso_real is not None:
        error_absoluto = abs(peso_promedio - peso_real)
        error_porcentual = (error_absoluto / peso_real) * 100
    
    print(f"\n[RESULTADO ENSEMBLE]")
    print(f"  Promedio: {peso_promedio:.2f} kg")
    print(f"  Rango: [{peso_min:.2f}, {peso_max:.2f}] kg")
    print(f"  Desv. Est.: {peso_std:.2f} kg")
    
    if peso_real is not None and error_absoluto is not None:
        print(f"\n[COMPARACIÓN CON PESO REAL]")
        print(f"  Peso Real: {peso_real:.2f} kg")
        print(f"  Error Absoluto: {error_absoluto:.2f} kg")
        print(f"  Error Porcentual: {error_porcentual:.2f}%")
        if error_absoluto <= 5:
            print(f"  ✅ Excelente (error ≤ 5 kg)")
        elif error_absoluto <= 8:
            print(f"  ⚠️  Aceptable (error ≤ 8 kg)")
        else:
            print(f"  ❌ Alto error (> 8 kg)")
    
    # Guardar predicciones individuales en JSON para análisis
    if save_debug:
        import json
        debug_data = {
            'archivo_original': filename,
            'id_vaca': id_vaca,
            'peso_real_kg': peso_real,
            'timestamp': timestamp,
            'peso_promedio_kg': round(peso_promedio, 2),
            'peso_min_kg': round(peso_min, 2),
            'peso_max_kg': round(peso_max, 2),
            'desviacion_estandar_kg': round(peso_std, 2),
            'error_absoluto_kg': round(error_absoluto, 2) if error_absoluto else None,
            'error_porcentual': round(error_porcentual, 2) if error_porcentual else None,
            'predicciones_por_fold': folds_info,
            'circulo_detectado': circ is not None,
            'circulo_coords': {'cx': int(circ[0]), 'cy': int(circ[1]), 'r': int(circ[2])} if circ else None
        }
        with open(DEBUG_DIR / f'{base_name}_predicciones.json', 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)
        print(f"[DEBUG] Predicciones guardadas en {base_name}_predicciones.json")
    
    # Generar advertencias según peso estimado
    advertencias = []
    
    if peso_promedio > 350:
        advertencias.append({
            'tipo': 'peso_alto',
            'mensaje': 'Precisión reducida en vacas pesadas',
            'icono': '⚠️',
            'detalle': 'El modelo tiene menor precisión en animales >350kg. Error esperado ~35%.'
        })
    
    if peso_promedio < 100:
        advertencias.append({
            'tipo': 'peso_bajo',
            'mensaje': 'Precisión reducida en terneros',
            'icono': '⚠️',
            'detalle': 'El modelo tiene menor precisión en terneros <100kg. Error esperado ~28%.'
        })
    
    if not circ:
        advertencias.append({
            'tipo': 'sin_circulo',
            'mensaje': 'Círculo de referencia NO detectado',
            'icono': '⚠️',
            'detalle': 'Sin el sticker de referencia, la precisión puede reducirse hasta un 10%.'
        })
    
    # Evaluar confianza según desviación estándar
    if peso_std < 5:
        confianza = 'alta'
        confianza_texto = 'Alta consistencia entre modelos'
    elif peso_std < 10:
        confianza = 'media'
        confianza_texto = 'Consistencia moderada entre modelos'
    else:
        confianza = 'baja'
        confianza_texto = 'Baja consistencia entre modelos'
        advertencias.append({
            'tipo': 'baja_confianza',
            'mensaje': 'Predicción con baja consistencia',
            'icono': '⚠️',
            'detalle': f'Alta variación entre modelos (±{peso_std:.1f}kg). Revisar calidad de imagen.'
        })
    
    # Preparar respuesta completa
    resultado = {
        'peso_estimado_kg': round(peso_promedio, 2),
        'confianza': confianza,
        'confianza_texto': confianza_texto,
        'circulo_detectado': circ is not None,
        'circulo_info': {
            'detectado': circ is not None,
            'centro': {'x': int(circ[0]), 'y': int(circ[1])} if circ else None,
            'radio': int(circ[2]) if circ else None,
            'mensaje': 'Círculo detectado correctamente' if circ else 'No se detectó el sticker de referencia'
        },
        'advertencias': advertencias,
        'detalles_ensemble': {
            'num_modelos': len(xgb_ensemble),
            'modelos_usados': [fold_idx for fold_idx, _ in xgb_ensemble],
            'peso_minimo': round(peso_min, 2),
            'peso_maximo': round(peso_max, 2),
            'desviacion_std': round(peso_std, 2),
            'rango_confianza_90': [
                round(peso_promedio - 1.645 * peso_std, 2),
                round(peso_promedio + 1.645 * peso_std, 2)
            ]
        }
    }
    
    print(f"\n[RESPUESTA AL FRONTEND]")
    print(f"  Peso estimado: {resultado['peso_estimado_kg']} kg")
    print(f"  Confianza: {resultado['confianza'].upper()}")
    print(f"  Círculo: {'✅ Detectado' if circ else '❌ NO detectado'}")
    print(f"  Advertencias: {len(advertencias)}")
    for adv in advertencias:
        print(f"    {adv['icono']} {adv['mensaje']}")
    
    return resultado
