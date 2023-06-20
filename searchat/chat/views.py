from django.http import HttpResponse
from rest_framework.views import APIView
from .chat import ChatBot


class ChatBotView(APIView):

    def get(self, request):
        request_params = request.GET
        q = request_params['q']
        result, sources = ChatBot.chat(q)
        result = q + '<br><br>' + result + '<br>' + '<br>'.join(sources)
        return HttpResponse(result)
