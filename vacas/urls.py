from django.urls import path
from .views import EstimarPesoVacaView

urlpatterns = [
    path('estimar-peso/', EstimarPesoVacaView.as_view(), name='estimar-peso'),
]