import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from config import Config
from model_service import SkinClassifier
from chat_service import SkincareChatbot

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# initialise services
classifier = SkinClassifier(
    model_path=Config.MODEL_PATH,
    image_size=Config.IMAGE_SIZE,
    class_names=Config.CLASS_NAMES
)

chatbot = SkincareChatbot(api_key=Config.GOOGLE_API_KEY)

# store active diagnoses per session
session_diagnoses = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'name': 'Lumina Skin Health API',
        'version': '1.0.0',
        'model': 'MobileNetV2 (Fine-Tuned)',
        'accuracy': '99.15%',
        'endpoints': {
            'POST /predict': 'Upload a skin image for classification',
            'POST /chat': 'Chat with the AI skincare assistant',
            'GET /health': 'API health check'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': classifier.model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided. Send an image file with key "image".'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Use: {Config.ALLOWED_EXTENSIONS}'}), 400

    try:
        image_bytes = file.read()
        result = classifier.predict(image_bytes)

        session_id = request.form.get('session_id', 'default')
        session_diagnoses[session_id] = result

        return jsonify({
            'success': True,
            'diagnosis': result
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/predict/base64', methods=['POST'])
def predict_base64():
    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided. Send base64 image with key "image".'}), 400

    try:
        import base64
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)

        result = classifier.predict(image_bytes)

        session_id = data.get('session_id', 'default')
        session_diagnoses[session_id] = result

        return jsonify({
            'success': True,
            'diagnosis': result
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided. Send JSON with key "message".'}), 400

    user_message = data['message']
    session_id = data.get('session_id', 'default')
    conversation_history = data.get('history', [])

    diagnosis = session_diagnoses.get(session_id) or data.get('diagnosis')

    try:
        response = chatbot.chat(
            user_message=user_message,
            diagnosis=diagnosis,
            conversation_history=conversation_history
        )

        return jsonify({
            'success': True,
            'response': response
        })

    except Exception as e:
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
