from django.urls import path

from . import views

urlpatterns = [
    path("", views.ChatBotView.as_view(), name="chat")
]
