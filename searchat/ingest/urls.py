from django.urls import path

from . import views

urlpatterns = [
    path("web_page/", views.WebIngestorView.as_view(), name="web_page")
]
