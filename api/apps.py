import os
from django.apps import AppConfig
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow import keras


class ApiConfig(AppConfig):
    #default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    MODEL_FILE = os.path.join(settings.MODELS, "modelcnn.h5")
    model = keras.models.load_model(MODEL_FILE)
