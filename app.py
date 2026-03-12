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

def split_by_original_paragraphs(text):
    """Split text by original newlines - keep original paragraph structure"""
    if not text:
        return []
    # Split by newlines only, keep empty lines as separators
    paragraphs = re.split(r'\n+', text)
    return [p.strip() for p in paragraphs if p.strip()]

def split_by_paragraphs(text, min_chunk_size=100):
    """Split text by newlines, merging small paragraphs into larger chunks"""
    if not text:
        return []

    # First split by newlines
    raw_paragraphs = re.split(r'\n+', text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    if not paragraphs:
        return []

    # Merge small paragraphs into larger chunks
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= min_chunk_size:
            # Add to current chunk
            if current_chunk:
                current_chunk += "\n" + para
            else:
                current_chunk = para
        else:
            # Current chunk is big enough, save it
            if current_chunk:
                chunks.append(current_chunk)

            # If this paragraph alone is bigger than min_chunk_size, use it directly
            if len(para) >= min_chunk_size:
                chunks.append(para)
                current_chunk = ""
            else:
                # Start new chunk with this paragraph
                current_chunk = para

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    # If we still have tiny chunks, merge them all
    if chunks and any(len(c) < min_chunk_size for c in chunks):
        merged = "\n".join(chunks)
        # Split into larger chunks
        words = merged.split()
        current = ""
        for word in words:
            if len(current) + len(word) + 1 <= min_chunk_size:
                current += " " + word if current else word
            else:
                if current:
                    chunks.append(current)
                current = word
        if current:
            chunks.append(current)

    return chunks

def split_by_sentences(text):
    """Split text by sentence endings (Chinese and English)"""
    if not text:
        return []
    # Split by Chinese and English sentence endings: 。！？.!?
    sentences = re.split(r'([。！？.!?])', text)
    result = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i]
        punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
        combined = (sentence + punctuation).strip()
        if combined:
            result.append(combined)
    # Handle case where there's no punctuation at end
    if sentences and sentences[-1].strip():
        result.append(sentences[-1].strip())
    return result

def detect_chunk(text):
    """Detect AIGC probability for a single chunk"""
    probability = detect_aigc(text)
    if probability is None:
        return None
    return round(probability, 4)

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

def read_file_content(file, filename):
    """Read file content with proper encoding handling.

    For TXT files: tries multiple encodings (utf-8, gbk, gb18030, utf-16, latin1)
    For DOCX files: uses python-docx to extract text

    Returns the text content or raises an exception if reading fails.
    """
    filename_lower = filename.lower()

    if filename_lower.endswith('.txt'):
        # Try multiple encodings for TXT files
        encodings = ['utf-8', 'gbk', 'gb18030', 'utf-16', 'latin1']

        # Read raw bytes first
        raw_content = file.read()

        for encoding in encodings:
            try:
                text = raw_content.decode(encoding)
                # Verify the decoded text is valid (not mostly replacement characters)
                if encoding != 'utf-8' and '�' in text:
                    # Check if too many replacement characters
                    replacement_ratio = text.count('�') / max(len(text), 1)
                    if replacement_ratio > 0.1:
                        continue
                logger.info(f"Successfully decoded {filename} with {encoding} encoding")
                return text
            except (UnicodeDecodeError, UnicodeError):
                continue

        # If all encodings fail, try with errors='ignore'
        return raw_content.decode('utf-8', errors='ignore')

    elif filename_lower.endswith('.docx'):
        # For DOCX files, use python-docx
        doc = Document(file)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text

    elif filename_lower.endswith('.pdf'):
        # For PDF files, use pypdf
        from pypdf import PdfReader
        pdf_reader = PdfReader(file)
        text_parts = []
        for page in pdf_reader.pages:
            text_parts.append(page.extract_text())
        return '\n'.join(text_parts)

    else:
        raise ValueError(f"Unsupported file format: {filename}")


@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '请选择文件'}), 400

        file = request.files['file']
        filename = file.filename

        # Use helper function to read file content with proper encoding
        text = read_file_content(file, filename)

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

@app.route('/api/detect-full', methods=['POST'])
def detect_full():
    """Detect full text with paragraph or sentence level chunking"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        mode = data.get('mode', 'paragraph')  # 'paragraph' or 'sentence'
        chunk_size = data.get('chunk_size', 'original')  # 'original' or number

        if not text or not text.strip():
            return jsonify({'success': False, 'error': '请输入要检测的文本'}), 400

        if chunk_size == 'original':
            # Split by original paragraphs (newlines only, no merging)
            chunks = split_by_original_paragraphs(text)
            mode = 'original'
        elif mode == 'paragraph':
            try:
                min_size = int(chunk_size) if chunk_size else 100
            except:
                min_size = 100
            chunks = split_by_paragraphs(text, min_chunk_size=min_size)
        elif mode == 'sentence':
            chunks = split_by_sentences(text)
        else:
            return jsonify({'success': False, 'error': '无效的mode参数，请使用"paragraph"或"sentence"'}), 400

        if not chunks:
            return jsonify({'success': False, 'error': '文本分割失败'}), 400

        results = []
        for i, chunk in enumerate(chunks):
            probability = detect_chunk(chunk)
            if probability is not None:
                results.append({
                    'index': i,
                    'text': chunk,
                    'probability': probability,
                    'text_length': len(chunk)
                })

        # Calculate weighted overall probability by text length
        total_chars = sum(r['text_length'] for r in results)
        overall_prob = sum(r['probability'] * r['text_length'] for r in results) / total_chars if total_chars > 0 else 0

        return jsonify({
            'success': True,
            'overall_probability': round(overall_prob, 4),
            'mode': mode,
            'chunks': results,
            'text_length': len(text)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/detect-chunk', methods=['POST'])
def detect_chunk_endpoint():
    """Detect single chunk text"""
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text or not text.strip():
            return jsonify({'success': False, 'error': '请输入要检测的文本'}), 400

        probability = detect_chunk(text)

        if probability is None:
            return jsonify({'success': False, 'error': '检测失败'}), 500

        return jsonify({
            'success': True,
            'probability': probability
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
