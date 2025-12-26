# VacaMetric - Backend (Modelos de IA)

## 🧠 Descripción
Backend desarrollado en Django con modelos de inteligencia artificial para la estimación del peso de ganado bovino mediante análisis de imágenes.

## 🚀 Repositorios del Proyecto
- **Frontend (App Móvil)**: https://github.com/JhosepSF/VacaMetric-Project-Front
- **Backend (Modelos IA)**: https://github.com/JhosepSF/VacaMetric-Project-Back

## 📋 Requisitos Previos

### Software Necesario
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Virtualenv (recomendado)

### Hardware Recomendado
- RAM: Mínimo 4GB (recomendado 8GB o más)
- Procesamiento: Los modelos requieren procesamiento intensivo
- GPU: Opcional pero mejora el rendimiento (CUDA compatible)

## 🔧 Instalación

### 1. Clonar el Repositorio
```bash
git clone https://github.com/JhosepSF/VacaMetric-Project-Back.git
cd VacaMetric-Project-Back
```

### 2. Crear Entorno Virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar Base de Datos
```bash
python manage.py migrate
```

### 5. Crear Superusuario (Opcional)
```bash
python manage.py createsuperuser
```

## ▶️ Ejecución

### Modo Desarrollo
```bash
# Servidor de desarrollo Django
python manage.py runserver 0.0.0.0:8000
```

### Modo Producción
```bash
# Con Gunicorn
gunicorn tesis_vacas_backend.wsgi:application --config gunicorn.conf.py
```

### Verificar Funcionamiento
Accede a: `http://localhost:8000/api/v1/vacas/health/`

Deberías ver: `{"status": "healthy"}`

## 📁 Estructura del Proyecto
```
Back/
├── tesis_vacas_backend/    # Configuración Django
│   ├── settings.py         # Configuraciones
│   ├── urls.py            # URLs principales
│   └── wsgi.py            # WSGI
├── vacas/                 # App principal
│   ├── views.py           # Endpoints API
│   ├── urls.py            # Rutas de la app
│   ├── models.py          # Modelos de datos
│   ├── health.py          # Health check
│   ├── artefactos_modelo/  # Modelos de IA
│   │   ├── yolov8x-seg.pt              # Segmentación YOLO
│   │   ├── backbone_wide_resnet50_2_state_dict.pt  # Backbone
│   │   ├── last_actual.pt              # Modelo PaDiM
│   │   └── xgboost_fold_*.ubj          # Modelos XGBoost (10 folds)
│   └── inference/         # Lógica de inferencia
│       ├── models_loader.py   # Carga de modelos
│       ├── preprocess.py      # Preprocesamiento
│       ├── backbone.py        # Extracción features
│       └── estimate.py        # Estimación final
├── manage.py
├── requirements.txt
└── db.sqlite3
```

## 🎯 API Endpoints

### Health Check
```
GET /api/v1/vacas/health/
```

### Estimación de Peso
```
POST /api/v1/vacas/estimar-peso/

Content-Type: multipart/form-data

Body:
- image: archivo de imagen (jpg, jpeg, png)

Response:
{
  "peso_estimado_kg": 450.5,
  "tiempo_procesamiento_ms": 1234
}
```

## 🤖 Modelos de IA Utilizados

### 1. YOLOv8x-seg
- **Función**: Segmentación de la vaca en la imagen
- **Archivo**: `yolov8x-seg.pt`

### 2. PaDiM (Wide ResNet50-2)
- **Función**: Extracción de características profundas
- **Archivos**: 
  - `backbone_wide_resnet50_2_state_dict.pt`
  - `last_actual.pt`

### 3. XGBoost (Ensemble de 10 modelos)
- **Función**: Predicción final del peso
- **Archivos**: `xgboost_fold_1.ubj` a `xgboost_fold_10.ubj`
- **Método**: Promedio de 10 modelos entrenados con validación cruzada

## 🛠️ Tecnologías Utilizadas

- **Django**: Framework web
- **Django REST Framework**: API REST
- **PyTorch**: Framework de deep learning
- **Ultralytics (YOLO)**: Detección y segmentación de objetos
- **XGBoost**: Gradient boosting
- **OpenCV**: Procesamiento de imágenes
- **NumPy**: Computación numérica
- **Pillow**: Manipulación de imágenes
- **Gunicorn**: Servidor WSGI para producción

## 📊 Flujo de Procesamiento

1. **Recepción**: La imagen se recibe vía POST
2. **Segmentación**: YOLOv8 segmenta la vaca
3. **Preprocesamiento**: Se recorta y normaliza la imagen
4. **Extracción**: PaDiM extrae características profundas
5. **Predicción**: 10 modelos XGBoost predicen el peso
6. **Ensemble**: Se promedian las predicciones
7. **Respuesta**: Se devuelve el peso estimado

## ⚙️ Configuración

### CORS (para desarrollo)
El backend está configurado para aceptar peticiones desde cualquier origen. En producción, edita `settings.py`:

```python
CORS_ALLOWED_ORIGINS = [
    "http://tu-dominio.com",
]
```

### Timeout
Las peticiones pueden tardar hasta 60 segundos debido al procesamiento intensivo de los modelos.

## 🐛 Solución de Problemas

### Error: "No module named 'torch'"
```bash
pip install torch torchvision
```

### Error: Memoria insuficiente
- Reduce el tamaño de las imágenes de entrada
- Cierra otras aplicaciones
- Considera usar un servidor con más RAM

### Error: Modelos no encontrados
Verifica que todos los archivos `.pt` y `.ubj` estén en `vacas/artefactos_modelo/`

## 🧪 Testing
```bash
python manage.py test
```

## 📈 Rendimiento
- Tiempo promedio de procesamiento: ~2-5 segundos
- Precisión del modelo: ~95% (según validación)
- Rango de peso soportado: 100-800 kg

## 👥 Autor
Miguel Angel Valles Coral

## 📄 Licencia
Este proyecto es parte de un trabajo académico.

## 📞 Soporte
Para más información, consulta el [Manual Técnico](MANUAL_TECNICO.md).
