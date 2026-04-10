"""
Run this in Google Colab to convert final_model.h5 to final_model.tflite.

Steps:
1. Open Google Colab
2. Mount your Google Drive
3. Run this script
4. Download the .tflite file and place it in backend/model/
"""
import tensorflow as tf

# Update this path to where your model is in Google Drive
MODEL_PATH = '/content/drive/MyDrive/Final Year project/models/final_model.h5'
OUTPUT_PATH = '/content/drive/MyDrive/Final Year project/models/final_model.tflite'

# Load the Keras model
model = tf.keras.models.load_model(MODEL_PATH)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TFLite model
with open(OUTPUT_PATH, 'wb') as f:
    f.write(tflite_model)

print(f'Converted successfully!')
print(f'Original .h5 size: {model.count_params()} parameters')
print(f'TFLite file size: {len(tflite_model) / (1024*1024):.1f} MB')
print(f'Saved to: {OUTPUT_PATH}')
