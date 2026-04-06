import numpy as np
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io


class SkinClassifier:
    def __init__(self, model_path, image_size, class_names):
        self.model = load_model(model_path)
        self.image_size = image_size
        self.class_names = class_names

    def preprocess_image(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((self.image_size, self.image_size))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def predict(self, image_bytes):
        processed = self.preprocess_image(image_bytes)
        predictions = self.model.predict(processed, verbose=0)[0]

        results = []
        for i, (cls, prob) in enumerate(zip(self.class_names, predictions)):
            results.append({
                'condition': cls,
                'confidence': round(float(prob) * 100, 2)
            })

        results.sort(key=lambda x: x['confidence'], reverse=True)

        return {
            'prediction': results[0]['condition'],
            'confidence': results[0]['confidence'],
            'all_predictions': results
        }
