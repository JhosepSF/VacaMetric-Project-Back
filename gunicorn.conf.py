# Gunicorn configuration for Render deployment
import os

# Binding - usa la variable de entorno PORT de Render
port = os.getenv('PORT', '10000')
bind = f"0.0.0.0:{port}"

# Worker configuration
workers = 1  # Solo 1 worker para ahorrar memoria (modelos ML son pesados)
worker_class = "sync"
timeout = 600  # 10 minutos para cargar modelos ML y procesar imágenes
graceful_timeout = 600
keepalive = 5

# Memory limits
worker_tmp_dir = "/dev/shm"  # Usar memoria compartida en lugar de disco

# Limits
max_requests = 50  # Reiniciar worker cada 50 requests para liberar memoria
max_requests_jitter = 10

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Preload app (carga Django antes de fork workers)
preload_app = False  # False para lazy loading de modelos
