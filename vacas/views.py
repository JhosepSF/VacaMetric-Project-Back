from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .inference.estimate import estimate_weight_from_bytes

class EstimarPesoVacaView(APIView):
    """
    POST /api/v1/vacas/estimar-peso/
    Body (form-data):
        - image: archivo de imagen (foto de la vaca + sticker de círculo)
    
    Response:
        {
            "peso_estimado_kg": float,
            "confianza": "alta|media|baja",
            "confianza_texto": str,
            "circulo_detectado": bool,
            "circulo_info": {...},
            "advertencias": [...],
            "detalles_ensemble": {...}
        }
    """
    def post(self, request, *args, **kwargs):
        print(f"\n{'='*60}")
        print(f"[VIEWS] Nueva petición POST recibida")
        print(f"[VIEWS] Headers: {dict(request.headers)}")
        print(f"[VIEWS] FILES: {request.FILES}")
        print(f"{'='*60}\n")
        
        img = request.FILES.get('image')
        if not img:
            print("[VIEWS] ERROR: No se recibió imagen")
            return Response(
                {"error": "Debe enviar una imagen en el campo 'image'."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        print(f"[VIEWS] Imagen recibida: {img.name}, tamaño: {img.size} bytes")
        
        try:
            resultado = estimate_weight_from_bytes(img)
            print(f"[VIEWS] Estimación exitosa: {resultado.get('peso_estimado_kg')} kg")
        except Exception as e:
            # Aquí puedes loguear el error con logging
            import traceback
            print("[VIEWS] ERROR durante estimación:")
            traceback.print_exc()
            return Response(
                {"error": f"Ocurrió un error al estimar el peso: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response(resultado, status=status.HTTP_200_OK)
