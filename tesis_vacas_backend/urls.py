from django.contrib import admin
from django.urls import path, include
from vacas.health import health_check

urlpatterns = [
    path('', health_check, name='health'),  # Health check en /
    path('health/', health_check, name='health-check'),  # También en /health/
    path('admin/', admin.site.urls),
    path('api/v1/vacas/', include('vacas.urls')), 
]
