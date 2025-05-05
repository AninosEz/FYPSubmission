from django.urls import path
from .views import predict_audio_view

urlpatterns = [
    path('predict/', predict_audio_view),
]
