from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import logging
import re
from docx import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def split_text_into_chunks(text, chunk_size):
    """将文本分割成块，保证句子完整"""
    sentences = re.split(r'([。！？\n])', text)
    chunks = []
    current_chunk = ""

    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i]
        punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
        combined = sentence + punctuation

        if len(current_chunk) + len(combined) <= chunk_size:
            current_chunk += combined
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(combined) > chunk_size:
                for j in range(0, len(combined), chunk_size):
                    chunks.append(combined[j:j+chunk_size])
                current_chunk = ""
            else:
                current_chunk = combined

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

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
        logger.info(f"Input text: {text[:50]}...")
        logger.info(f"Raw logits: {logits}")
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)
        logger.info(f"Probabilities: {probs}")
        # Model trained with: label 0 = AI-generated, label 1 = human-written
        # So index 0 is AI-generated probability
        ai_prob = probs[0][0].item()
        logger.info(f"AI probability: {ai_prob}")

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

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '请选择文件'}), 400

        file = request.files['file']
        filename = file.filename.lower()

        if filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        elif filename.endswith('.docx'):
            doc = Document(file)
            text = '\n'.join([para.text for para in doc.paragraphs])
        else:
            return jsonify({'success': False, 'error': '不支持的文件格式'}), 400

        return jsonify({'success': True, 'text': text, 'filename': file.filename})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/detect-chunks', methods=['POST'])
def detect_chunks():
    try:
        data = request.get_json()
        text = data.get('text', '')
        chunk_size = data.get('chunk_size', 200)

        if not text or not text.strip():
            return jsonify({'success': False, 'error': '请输入要检测的文本'}), 400

        chunks = split_text_into_chunks(text, chunk_size)
        results = []

        for i, chunk in enumerate(chunks):
            inputs = tokenizer(chunk, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                ai_prob = probs[0][0].item()

            results.append({
                'index': i,
                'text': chunk,
                'probability': round(ai_prob, 4),
                'text_length': len(chunk)
            })

        total_chars = sum(r['text_length'] for r in results)
        overall_prob = sum(r['probability'] * r['text_length'] for r in results) / total_chars if total_chars > 0 else 0

        return jsonify({
            'success': True,
            'overall_probability': round(overall_prob, 4),
            'total_chunks': len(chunks),
            'chunks': results,
            'text_length': len(text)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
