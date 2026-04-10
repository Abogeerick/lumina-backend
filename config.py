import os

class Config:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'final_model.tflite')
    IMAGE_SIZE = 224
    CLASS_NAMES = ['Vitiligo', 'acne', 'hyperpigmentation']
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max upload
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
