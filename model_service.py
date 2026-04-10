import numpy as np
from PIL import Image
import io
import tflite_runtime.interpreter as tflite


class SkinClassifier:
    def __init__(self, model_path, image_size, class_names):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.image_size = image_size
        self.class_names = class_names
        self.model = True  # flag for health check compatibility

    def preprocess_image(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((self.image_size, self.image_size))
        img_array = np.array(img, dtype=np.float32)
        # MobileNetV2 preprocessing: scale [0, 255] to [-1, 1]
        img_array = (img_array / 127.5) - 1.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_bytes):
        processed = self.preprocess_image(image_bytes)
        self.interpreter.set_tensor(self.input_details[0]['index'], processed)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

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
