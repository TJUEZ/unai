from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

app = Flask(__name__)
CORS(app)

# Model configuration
MODEL_PATH = "."
TOKENIZER_PATH = "."
MAX_LENGTH = 512

# Global variables for model and tokenizer
tokenizer = None
model = None

def load_model():
    """Load the AIGC detection model"""
    global tokenizer, model

    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

    print("Loading model...")
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("Model loaded successfully!")

def detect_aigc(text):
    """Detect if text is AI-generated"""
    if not text or not text.strip():
        return None

    # Tokenize
    inputs = tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)
        # Assuming index 1 is AI-generated probability
        ai_prob = probs[0][1].item()

    return ai_prob

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text or not text.strip():
            return jsonify({
                'success': False,
                'error': '请输入要检测的文本'
            }), 400

        probability = detect_aigc(text)

        if probability is None:
            return jsonify({
                'success': False,
                'error': '检测失败'
            }), 500

        # Determine result label
        if probability > 0.5:
            result = "AI生成的文本"
        else:
            result = "人类撰写的文本"

        return jsonify({
            'success': True,
            'probability': round(probability, 4),
            'result': result,
            'text_length': len(text)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
