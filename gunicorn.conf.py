# Gunicorn configuration for Render deployment
import multiprocessing

# Binding
bind = "0.0.0.0:$PORT"

# Worker configuration
workers = 1  # Solo 1 worker para ahorrar memoria (modelos ML son pesados)
worker_class = "sync"
timeout = 300  # 5 minutos para cargar modelos ML
graceful_timeout = 300
keepalive = 5

# Limits
max_requests = 100
max_requests_jitter = 10

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
