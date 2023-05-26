from django.urls import path
from .views import *

urlpatterns = [
    path('api/upload', UploadView.as_view(), name = 'prediction'),
]