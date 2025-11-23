from django.apps import AppConfig


class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'

    def ready(self):
        # Import the predictor loading function here to ensure models are loaded on startup
        from .services.predict import load_predictor
        load_predictor()
