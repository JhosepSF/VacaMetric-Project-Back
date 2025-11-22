from django.http import JsonResponse

def health_check(request):
    """Health check endpoint para Render"""
    return JsonResponse({"status": "ok", "service": "VacaMetric API"})
