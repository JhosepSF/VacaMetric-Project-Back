# Manual Técnico - VacaMetric

## 📚 Índice
1. [Introducción](#introducción)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Componentes del Frontend](#componentes-del-frontend)
4. [Componentes del Backend](#componentes-del-backend)
5. [Modelos de Inteligencia Artificial](#modelos-de-inteligencia-artificial)
6. [API REST](#api-rest)
7. [Base de Datos](#base-de-datos)
8. [Instalación y Configuración](#instalación-y-configuración)
9. [Despliegue](#despliegue)
10. [Mantenimiento](#mantenimiento)

---

## 1. Introducción

### 1.1 Propósito del Documento
Este manual técnico describe la arquitectura, componentes, tecnologías y procesos de desarrollo del sistema VacaMetric, una aplicación móvil para la estimación del peso de ganado bovino mediante inteligencia artificial.

### 1.2 Alcance
El sistema está compuesto por:
- **Frontend**: Aplicación móvil desarrollada en React Native con Expo
- **Backend**: API REST desarrollada en Django con modelos de IA

### 1.3 Audiencia
Este documento está dirigido a:
- Desarrolladores que darán mantenimiento al sistema
- Administradores de sistemas
- Personal técnico encargado del despliegue

---

## 2. Arquitectura del Sistema

### 2.1 Arquitectura General

```
┌─────────────────────┐
│   Aplicación Móvil  │
│   (React Native)    │
└──────────┬──────────┘
           │
           │ HTTP/REST
           │
┌──────────▼──────────┐
│    API Backend      │
│     (Django)        │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
┌────▼────┐ ┌───▼────┐
│ SQLite  │ │ Modelos│
│   DB    │ │   IA   │
└─────────┘ └────────┘
```

### 2.2 Patrón de Arquitectura
- **Frontend**: Arquitectura basada en componentes (Component-Based)
- **Backend**: Arquitectura MVT (Model-View-Template) de Django
- **Comunicación**: REST API con JSON

### 2.3 Flujo de Datos

```
Usuario → Captura Imagen → App Móvil → API REST → Backend
                                                      ↓
                                            Procesamiento IA
                                                      ↓
Usuario ← Muestra Peso ← App Móvil ← Respuesta JSON ← Backend
```

---

## 3. Componentes del Frontend

### 3.1 Tecnologías

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| React Native | Latest | Framework móvil |
| Expo | SDK 49+ | Plataforma de desarrollo |
| TypeScript | 4.9+ | Lenguaje de programación |
| Axios | Latest | Cliente HTTP |
| AsyncStorage | Latest | Persistencia local |
| React Navigation | Latest | Navegación |

### 3.2 Estructura de Carpetas

```
Front/
├── src/
│   ├── components/              # Componentes reutilizables
│   │   ├── CustomHeader.tsx     # Encabezado personalizado
│   │   └── Footer.tsx           # Pie de página
│   ├── navigation/              # Sistema de navegación
│   │   └── AppNavigator.tsx     # Navegador principal
│   ├── screens/                 # Pantallas de la app
│   │   ├── TomarFotoVacaScreen.tsx    # Captura de imagen
│   │   ├── MenuRegistroScreen.tsx     # Menú principal
│   │   └── HistorialScreen.tsx        # Historial
│   └── services/                # Servicios externos
│       └── api.ts               # Cliente API
├── assets/                      # Recursos estáticos
├── App.tsx                      # Punto de entrada
└── package.json                 # Dependencias
```

### 3.3 Componentes Principales

#### 3.3.1 AppNavigator.tsx
**Responsabilidad**: Gestión de navegación entre pantallas

```typescript
- Stack Navigator
- Configuración de rutas
- Transiciones entre pantallas
```

#### 3.3.2 TomarFotoVacaScreen.tsx
**Responsabilidad**: Captura y procesamiento de imágenes

**Funcionalidades**:
- Solicitud de permisos de cámara
- Captura de foto
- Envío a API
- Manejo de estados (cargando, error, éxito)

#### 3.3.3 api.ts
**Responsabilidad**: Comunicación con el backend

**Métodos**:
- `estimarPesoVaca(fotoUri: string): Promise<number>`
- `guardarEnHistorial(fotoUri: string, peso: number): Promise<void>`

**Configuración**:
```typescript
const api = axios.create({
  baseURL: 'http://192.168.100.7:8000',
  timeout: 60000, // 60 segundos
});
```

### 3.4 Almacenamiento Local

**AsyncStorage**:
- Clave: `historial_estimaciones`
- Formato: JSON Array
- Límite: 50 registros más recientes

**Estructura de datos**:
```json
{
  "id": "1703508123456",
  "fotoUri": "file:///path/to/image.jpg",
  "pesoEstimado": 450.5,
  "fecha": "2024-12-25T10:30:00.000Z"
}
```

---

## 4. Componentes del Backend

### 4.1 Tecnologías

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| Python | 3.8+ | Lenguaje base |
| Django | 3.2+ | Framework web |
| PyTorch | Latest | Deep Learning |
| Ultralytics | Latest | YOLO |
| XGBoost | Latest | ML |
| Gunicorn | Latest | Servidor WSGI |

### 4.2 Estructura de Carpetas

```
Back/
├── tesis_vacas_backend/         # Proyecto Django
│   ├── settings.py              # Configuración
│   ├── urls.py                  # URLs raíz
│   ├── wsgi.py                  # WSGI
│   └── asgi.py                  # ASGI
├── vacas/                       # App principal
│   ├── views.py                 # Vistas/Controllers
│   ├── urls.py                  # URLs de la app
│   ├── models.py                # Modelos de datos
│   ├── health.py                # Health check
│   ├── artefactos_modelo/       # Archivos de modelos
│   │   ├── yolov8x-seg.pt
│   │   ├── backbone_wide_resnet50_2_state_dict.pt
│   │   ├── last_actual.pt
│   │   └── xgboost_fold_*.ubj
│   └── inference/               # Lógica de IA
│       ├── models_loader.py
│       ├── preprocess.py
│       ├── backbone.py
│       └── estimate.py
├── manage.py
├── requirements.txt
└── db.sqlite3
```

### 4.3 Módulos de Inferencia

#### 4.3.1 models_loader.py
**Responsabilidad**: Carga de modelos de IA en memoria

```python
Clases:
- ModelLoader: Singleton para cargar modelos una sola vez

Modelos cargados:
- YOLO: Segmentación
- PaDiM: Extracción de features
- XGBoost: 10 modelos de predicción
```

#### 4.3.2 preprocess.py
**Responsabilidad**: Preprocesamiento de imágenes

**Funciones**:
```python
- segment_cow(image_path): Segmenta la vaca usando YOLO
- crop_to_mask(image, mask): Recorta imagen a la máscara
- resize_image(image, size): Redimensiona imagen
```

#### 4.3.3 backbone.py
**Responsabilidad**: Extracción de características

**Proceso**:
1. Carga backbone Wide ResNet50-2
2. Aplica PaDiM para extracción multi-escala
3. Retorna vector de características (embedding)

#### 4.3.4 estimate.py
**Responsabilidad**: Estimación final del peso

**Algoritmo**:
```python
1. Recibe embedding de features
2. Aplica 10 modelos XGBoost
3. Promedia las predicciones
4. Retorna peso estimado en kg
```

### 4.4 Vista Principal

#### views.py - EstimarPesoView

```python
class EstimarPesoView(APIView):
    """
    POST /api/v1/vacas/estimar-peso/
    
    Input: multipart/form-data con imagen
    Output: JSON con peso_estimado_kg
    """
```

**Flujo de procesamiento**:
1. Validación de imagen recibida
2. Guardado temporal de imagen
3. Segmentación con YOLO
4. Preprocesamiento
5. Extracción de features con PaDiM
6. Predicción con XGBoost ensemble
7. Limpieza de archivos temporales
8. Retorno de resultado

---

## 5. Modelos de Inteligencia Artificial

### 5.1 Pipeline de Procesamiento

```
Imagen → YOLO → Mask → Crop → Resize → PaDiM → Features → XGBoost → Peso
```

### 5.2 YOLOv8x-seg

**Tipo**: Modelo de segmentación semántica

**Características**:
- Arquitectura: YOLOv8 Extra Large
- Tarea: Segmentación de instancias
- Input: Imagen RGB
- Output: Máscara binaria de la vaca

**Parámetros**:
```python
modelo.predict(
    imagen,
    conf=0.25,      # Confianza mínima
    iou=0.45,       # IoU threshold
    classes=[19]    # Clase 'cow' en COCO
)
```

### 5.3 PaDiM (Patch Distribution Modeling)

**Tipo**: Modelo de detección de anomalías adaptado para features

**Arquitectura**:
- Backbone: Wide ResNet50-2 pre-entrenado en ImageNet
- Capas utilizadas: layer1, layer2, layer3
- Dimensión de features: 1792 canales

**Proceso**:
```python
1. Forward pass por Wide ResNet50-2
2. Extracción de features multi-escala
3. Pooling adaptativo (AdaptiveAvgPool2d)
4. Concatenación de features
5. Embedding final de 1792 dimensiones
```

### 5.4 XGBoost Ensemble

**Tipo**: Gradient Boosting Machine

**Configuración**:
- Número de modelos: 10 (validación cruzada 10-fold)
- Parámetros principales:
  ```python
  {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'reg:squarederror'
  }
  ```

**Método de ensemble**:
- Promedio simple de 10 predicciones
- Reduce varianza y mejora generalización

**Input**: Vector de 1792 características
**Output**: Peso en kilogramos

### 5.5 Métricas de Rendimiento

| Modelo | Métrica | Valor |
|--------|---------|-------|
| YOLO | mAP@0.5 | ~0.85 |
| PaDiM | Feature dimension | 1792 |
| XGBoost | MAE | ~15 kg |
| XGBoost | RMSE | ~20 kg |
| XGBoost | R² | ~0.95 |

---

## 6. API REST

### 6.1 Base URL
```
http://<servidor>:8000/api/v1/vacas/
```

### 6.2 Endpoints

#### 6.2.1 Health Check
```http
GET /api/v1/vacas/health/

Response 200:
{
  "status": "healthy"
}
```

#### 6.2.2 Estimación de Peso
```http
POST /api/v1/vacas/estimar-peso/

Headers:
Content-Type: multipart/form-data

Body:
- image: File (JPEG, PNG)

Response 200:
{
  "peso_estimado_kg": 450.5,
  "tiempo_procesamiento_ms": 2345
}

Response 400:
{
  "error": "No se detectó ninguna vaca en la imagen"
}

Response 500:
{
  "error": "Error interno del servidor"
}
```

### 6.3 Códigos de Estado

| Código | Significado |
|--------|-------------|
| 200 | Éxito |
| 400 | Solicitud incorrecta |
| 404 | Recurso no encontrado |
| 500 | Error del servidor |
| 503 | Servicio no disponible |

### 6.4 Manejo de Errores

**Errores comunes**:
```python
- "No se proporcionó una imagen": Falta campo 'image'
- "No se detectó ninguna vaca": YOLO no encontró vaca
- "Error en el procesamiento": Fallo en pipeline
```

---

## 7. Base de Datos

### 7.1 Motor
SQLite3 (Desarrollo y producción ligera)

### 7.2 Esquema

Actualmente el sistema no persiste las estimaciones en la base de datos del backend, solo utiliza las tablas por defecto de Django:

```sql
- django_migrations
- auth_user
- auth_group
- django_session
- django_content_type
```

### 7.3 Extensibilidad

Para guardar estimaciones en el futuro, se puede crear el modelo:

```python
class Estimacion(models.Model):
    imagen = models.ImageField(upload_to='estimaciones/')
    peso_estimado = models.FloatField()
    fecha_creacion = models.DateTimeField(auto_now_add=True)
    usuario = models.ForeignKey(User, on_delete=models.CASCADE)
```

---

## 8. Instalación y Configuración

### 8.1 Requisitos del Sistema

**Frontend**:
- Node.js 16+
- npm o yarn
- 2 GB RAM mínimo
- Conexión a internet

**Backend**:
- Python 3.8+
- 4 GB RAM mínimo (8 GB recomendado)
- 2 GB espacio en disco
- CPU multi-core (GPU opcional)

### 8.2 Instalación Frontend

```bash
# Clonar repositorio
git clone https://github.com/JhosepSF/VacaMetric-Project-Front.git
cd VacaMetric-Project-Front

# Instalar dependencias
npm install

# Configurar API URL en src/services/api.ts
# baseURL: 'http://<IP_BACKEND>:8000'

# Ejecutar
npx expo start
```

### 8.3 Instalación Backend

```bash
# Clonar repositorio
git clone https://github.com/JhosepSF/VacaMetric-Project-Back.git
cd VacaMetric-Project-Back

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Migrar base de datos
python manage.py migrate

# Ejecutar servidor
python manage.py runserver 0.0.0.0:8000
```

### 8.4 Variables de Entorno

#### Backend - settings.py

```python
DEBUG = True  # False en producción
ALLOWED_HOSTS = ['*']  # Restringir en producción
CORS_ALLOW_ALL_ORIGINS = True  # Restringir en producción

# Configuración de archivos estáticos
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'
```

---

## 9. Despliegue

### 9.1 Despliegue Frontend

#### Opción 1: Expo Go (Desarrollo)
```bash
npx expo start
# Escanear QR con Expo Go
```

#### Opción 2: Build APK (Producción)
```bash
# Instalar EAS CLI
npm install -g eas-cli

# Login
eas login

# Configurar
eas build:configure

# Build
eas build -p android --profile production
```

### 9.2 Despliegue Backend

#### Opción 1: Desarrollo
```bash
python manage.py runserver 0.0.0.0:8000
```

#### Opción 2: Producción con Gunicorn
```bash
# Instalar gunicorn
pip install gunicorn

# Ejecutar
gunicorn tesis_vacas_backend.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers 2 \
    --timeout 120
```

#### Opción 3: Producción con Nginx + Gunicorn

**gunicorn.conf.py**:
```python
bind = '0.0.0.0:8000'
workers = 2
timeout = 120
worker_class = 'sync'
```

**Nginx**:
```nginx
server {
    listen 80;
    server_name tu-dominio.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /media/ {
        alias /path/to/media/;
    }
}
```

### 9.3 Consideraciones de Producción

1. **Seguridad**:
   - Cambiar `DEBUG = False`
   - Configurar `ALLOWED_HOSTS`
   - Usar HTTPS
   - Restringir CORS

2. **Rendimiento**:
   - Usar Gunicorn con múltiples workers
   - Configurar cache
   - Optimizar tamaño de imágenes
   - Considerar GPU para inferencia

3. **Monitoreo**:
   - Logs centralizados
   - Métricas de rendimiento
   - Alertas de errores

---

## 10. Mantenimiento

### 10.1 Actualización de Modelos

Para actualizar modelos de IA:

1. Entrenar nuevo modelo
2. Guardar en formato correspondiente (.pt, .ubj)
3. Reemplazar archivo en `artefactos_modelo/`
4. Reiniciar servidor backend

### 10.2 Logs

**Backend**:
```python
# Django logs en consola
# Configurar logging en settings.py para archivo
```

**Frontend**:
```javascript
console.log('[API] mensaje');
// Logs en consola del depurador
```

### 10.3 Backup

**Base de datos**:
```bash
# Backup SQLite
cp db.sqlite3 db.sqlite3.backup

# Restaurar
cp db.sqlite3.backup db.sqlite3
```

**Modelos**:
```bash
# Backup carpeta artefactos
tar -czf artefactos_backup.tar.gz vacas/artefactos_modelo/
```

### 10.4 Troubleshooting

**Problema**: Timeout en estimaciones
**Solución**: Aumentar timeout en frontend y backend

**Problema**: Memoria insuficiente
**Solución**: Reducir workers, optimizar modelos

**Problema**: No detecta vaca
**Solución**: Mejorar calidad de imagen, ajustar confianza YOLO

---

## Apéndices

### A. Dependencias Frontend

Ver `package.json` para lista completa

### B. Dependencias Backend

Ver `requirements.txt` para lista completa

### C. Diagramas Adicionales

_(Pueden agregarse diagramas UML, secuencia, etc.)_

---

**Versión**: 1.0  
**Fecha**: Diciembre 2024  
**Autor**: Jhosep SF
