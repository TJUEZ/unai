from flask import Flask, request, jsonify, render_template, send_from_directory
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
    """Split text by original newlines - keep original paragraph structure

    Returns list of tuples: (start_pos, chunk_text)
    """
    if not text:
        return []

    # Split by newlines only, keep empty lines as separators
    paragraphs = re.split(r'\n+', text)

    result = []
    current_pos = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            # Skip empty paragraphs but track position
            current_pos += len(para) + 1  # +1 for newline
            continue

        # Find the start position in original text
        start_pos = text.find(para, current_pos)
        if start_pos == -1:
            start_pos = current_pos

        result.append((start_pos, para))
        current_pos = start_pos + len(para) + 1

    return result

def split_by_paragraphs(text, min_chunk_size=100):
    """Split text into chunks, preserving paragraphs and sentence boundaries

    Returns list of tuples: (start_pos, chunk_text)
    """
    if not text:
        return []

    # Split by double newlines to get paragraphs
    paragraphs_info = []
    current_pos = 0

    # Find all paragraphs with their positions
    para_pattern = re.compile(r'\n\s*\n')
    last_end = 0

    for match in para_pattern.finditer(text):
        para_text = text[last_end:match.start()].strip()
        if para_text:
            start = text.find(para_text, last_end)
            if start == -1:
                start = last_end
            paragraphs_info.append((start, para_text))
        last_end = match.end()

    # Handle remaining text after last double newline
    remaining = text[last_end:].strip()
    if remaining:
        start = text.find(remaining, last_end)
        if start == -1:
            start = last_end
        paragraphs_info.append((start, remaining))

    # If no double newlines, try single newlines
    if not paragraphs_info:
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                start = text.find(line, current_pos)
                if start == -1:
                    start = current_pos
                paragraphs_info.append((start, line))
                current_pos = start + len(line)

    if not paragraphs_info:
        return []

    # If total text is smaller than 1.3x target, don't split
    total_len = sum(len(p[1]) for p in paragraphs_info)
    if total_len < min_chunk_size * 1.3:
        return paragraphs_info

    chunks = []

    for para_start, para in paragraphs_info:
        para = para.strip()
        if not para:
            continue

        # Skip paragraphs smaller than half the target - keep as-is
        if len(para) < min_chunk_size * 0.5:
            chunks.append((para_start, para))
            continue

        # Split paragraph into sentences (preserving punctuation)
        sentences = re.split(r'(?<=[。！？.!?])\s*', para)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            sentences = [para]

        # Group sentences into chunks targeting min_chunk_size
        current_chunk = ""
        current_start = para_start
        for sent in sentences:
            if not current_chunk:
                current_chunk = sent
                # Find start position of this sentence in original text
                sent_start = para.find(sent)
                if sent_start >= 0:
                    current_start = para_start + sent_start
            elif len(current_chunk) + len(sent) <= min_chunk_size:
                current_chunk += " " + sent
            elif len(sent) >= min_chunk_size:
                if current_chunk:
                    chunks.append((current_start, current_chunk))
                current_chunk = sent
                sent_start = para.find(sent)
                if sent_start >= 0:
                    current_start = para_start + sent_start
            else:
                if len(current_chunk) + len(sent) <= min_chunk_size * 1.15:
                    current_chunk += " " + sent
                else:
                    chunks.append((current_start, current_chunk))
                    current_chunk = sent
                    sent_start = para.find(sent)
                    if sent_start >= 0:
                        current_start = para_start + sent_start

        if current_chunk:
            chunks.append((current_start, current_chunk))

    # Merge small chunks
    small_threshold = min_chunk_size * 0.5

    result = []
    i = 0
    while i < len(chunks):
        chunk_start, chunk = chunks[i]

        if len(chunk) < small_threshold:
            # Try merge with previous
            if result and len(result[-1][1]) + len(chunk) <= min_chunk_size * 1.4:
                prev_start, prev_text = result[-1]
                result[-1] = (prev_start, prev_text + " " + chunk)
                i += 1
                continue
            # Try merge with next
            if i + 1 < len(chunks) and len(chunk) + len(chunks[i + 1][1]) <= min_chunk_size * 1.4:
                next_start, next_text = chunks[i + 1]
                result.append((chunk_start, chunk + " " + next_text))
                i += 2
                continue

        result.append((chunk_start, chunk))
        i += 1

    return result

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
    # Only serve React frontend
    frontend_build = os.path.join(os.path.dirname(__file__), 'frontend', 'dist', 'index.html')
    if os.path.exists(frontend_build):
        return send_from_directory(os.path.join(os.path.dirname(__file__), 'frontend', 'dist'), 'index.html')
    return jsonify({'error': 'Frontend not found. Please run npm run build in frontend directory.'}), 404

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from frontend build"""
    frontend_dist = os.path.join(os.path.dirname(__file__), 'frontend', 'dist')
    if os.path.exists(frontend_dist):
        return send_from_directory(frontend_dist, filename)
    return jsonify({'error': 'Static file not found'}), 404

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
        chunk_size = data.get('chunk_size', 'original')  # 'original' or number

        if not text or not text.strip():
            return jsonify({'success': False, 'error': '请输入要检测的文本'}), 400

        # Determine mode and chunk size
        if chunk_size == 'original' or chunk_size == 'original':
            # Split by original paragraphs (newlines only, no merging)
            chunks = split_by_original_paragraphs(text)
            mode = 'original'
        else:
            # Try to parse as number
            try:
                min_size = int(chunk_size) if chunk_size else 100
            except (ValueError, TypeError):
                min_size = 100
            chunks = split_by_paragraphs(text, min_chunk_size=min_size)
            mode = 'paragraph'

        if not chunks:
            return jsonify({'success': False, 'error': '文本分割失败'}), 400

        results = []
        for i, chunk_data in enumerate(chunks):
            # Handle both old format (just text) and new format (start_pos, text)
            if isinstance(chunk_data, tuple):
                start_pos, chunk_text = chunk_data
            else:
                start_pos = 0
                chunk_text = chunk_data

            probability = detect_chunk(chunk_text)
            if probability is not None:
                results.append({
                    'index': i,
                    'text': chunk_text,
                    'probability': probability,
                    'text_length': len(chunk_text),
                    'start_pos': start_pos,
                    'end_pos': start_pos + len(chunk_text)  # 添加结束位置
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
