from django.http import HttpResponse
from .ingestors import WebIngestor
from rest_framework.views import APIView


class WebIngestorView(APIView):

    def get(self, request):
        request_params = request.GET
        url = request_params['url']
        result = WebIngestor.ingest_webpage(url)
        return HttpResponse(result)
