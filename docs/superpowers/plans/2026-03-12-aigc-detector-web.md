# AIGC Text Detector Web Interface - Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Flask web application that runs the Chinese AIGC detection model locally with a clean, modern web interface.

**Architecture:** Flask backend serves the PyTorch model via REST API. Frontend is a single HTML page with vanilla JS communicating with the Flask API.

**Tech Stack:** Flask, Flask-CORS, Transformers, PyTorch, HTML/CSS/JS

---

## File Structure

```
AIGC_detector_zhv3/
├── app.py                      # Flask application (NEW)
├── requirements_web.txt        # Python dependencies (NEW)
├── templates/
│   └── index.html              # Frontend (NEW)
├── pytorch_model.bin           # Existing model weights
├── config.json                 # Existing model config
├── tokenizer_config.json       # Existing tokenizer config
├── vocab.txt                   # Existing vocabulary
├── special_tokens_map.json     # Existing special tokens
└── AIGC_text_detector/         # Original detector code (read-only)
```

---

## Implementation Tasks

### Task 1: Create Python Dependencies File

**Files:**
- Create: `requirements_web.txt`

- [ ] **Step 1: Create requirements file**

```
flask>=2.0
flask-cors>=3.0
torch>=1.10
transformers==4.27
```

- [ ] **Step 2: Commit**

---

### Task 2: Create Flask Backend

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write Flask app with model loading and API**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: add Flask backend for AIGC detection"
```

---

### Task 3: Create Frontend HTML

**Files:**
- Create: `templates/index.html`

- [ ] **Step 1: Write frontend HTML with Clean Modern design**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIGC文本检测器</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
            background: #f5f7fa;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            padding: 40px 20px;
        }

        .container {
            width: 100%;
            max-width: 800px;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 28px;
            color: #2c3e50;
            margin-bottom: 8px;
        }

        .header p {
            color: #7f8c8d;
            font-size: 14px;
        }

        .card {
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
            padding: 30px;
            margin-bottom: 20px;
        }

        .input-label {
            display: block;
            font-size: 16px;
            color: #2c3e50;
            margin-bottom: 12px;
            font-weight: 500;
        }

        textarea {
            width: 100%;
            height: 200px;
            padding: 15px;
            border: 2px solid #e1e8ef;
            border-radius: 8px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s;
        }

        textarea:focus {
            outline: none;
            border-color: #4A90E2;
        }

        textarea::placeholder {
            color: #b0bec5;
        }

        .btn-detect {
            display: block;
            width: 100%;
            padding: 15px;
            margin-top: 20px;
            background: #4A90E2;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.3s;
        }

        .btn-detect:hover {
            background: #357ABD;
        }

        .btn-detect:disabled {
            background: #b0bec5;
            cursor: not-allowed;
        }

        .result-card {
            display: none;
        }

        .result-card.show {
            display: block;
        }

        .result-header {
            text-align: center;
            margin-bottom: 25px;
        }

        .result-label {
            font-size: 18px;
            color: #7f8c8d;
            margin-bottom: 10px;
        }

        .result-percentage {
            font-size: 48px;
            font-weight: bold;
            color: #2c3e50;
        }

        .result-text {
            text-align: center;
            font-size: 20px;
            font-weight: 500;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .result-text.ai {
            background: #fdf2f2;
            color: #e74c3c;
            border: 1px solid #fadbd8;
        }

        .result-text.human {
            background: #f2fdf5;
            color: #27ae60;
            border: 1px solid #d5f4e3;
        }

        .progress-bar {
            height: 12px;
            background: #e1e8ef;
            border-radius: 6px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.5s ease, background 0.5s ease;
        }

        .progress-labels {
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            font-size: 12px;
            color: #7f8c8d;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .loading.show {
            display: block;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #e1e8ef;
            border-top-color: #4A90E2;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .error-message {
            display: none;
            background: #fdf2f2;
            color: #e74c3c;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            text-align: center;
        }

        .error-message.show {
            display: block;
        }

        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #e1e8ef;
            font-size: 14px;
            color: #7f8c8d;
        }

        .info-row:last-child {
            border-bottom: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AIGC文本检测器</h1>
            <p>基于深度学习的中文AI生成文本检测工具</p>
        </div>

        <div class="card">
            <label class="input-label" for="text-input">请输入要检测的文本：</label>
            <textarea id="text-input" placeholder="在此粘贴或输入中文文本..."></textarea>
            <button class="btn-detect" id="detect-btn" onclick="detectText()">检测</button>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>正在分析文本...</p>
            </div>

            <div class="error-message" id="error-message"></div>
        </div>

        <div class="card result-card" id="result-card">
            <div class="result-header">
                <div class="result-label">AI生成概率</div>
                <div class="result-percentage" id="result-percentage">0%</div>
            </div>
            <div class="result-text" id="result-text"></div>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill"></div>
            </div>
            <div class="progress-labels">
                <span>人类撰写</span>
                <span>AI生成</span>
            </div>
            <div class="info-row" style="margin-top: 20px;">
                <span>文本长度</span>
                <span id="text-length">0 字符</span>
            </div>
        </div>
    </div>

    <script>
        async function detectText() {
            const text = document.getElementById('text-input').value.trim();
            const btn = document.getElementById('detect-btn');
            const loading = document.getElementById('loading');
            const errorMsg = document.getElementById('error-message');
            const resultCard = document.getElementById('result-card');

            // Reset UI
            errorMsg.classList.remove('show');
            resultCard.classList.remove('show');

            if (!text) {
                showError('请输入要检测的文本');
                return;
            }

            // Show loading
            btn.disabled = true;
            loading.classList.add('show');

            try {
                const response = await fetch('/api/detect', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text })
                });

                const data = await response.json();

                if (!data.success) {
                    showError(data.error || '检测失败，请重试');
                    return;
                }

                // Display result
                displayResult(data);

            } catch (error) {
                showError('网络错误，请检查服务器是否运行');
                console.error(error);
            } finally {
                btn.disabled = false;
                loading.classList.remove('show');
            }
        }

        function displayResult(data) {
            const resultCard = document.getElementById('result-card');
            const percentage = document.getElementById('result-percentage');
            const resultText = document.getElementById('result-text');
            const progressFill = document.getElementById('progress-fill');
            const textLength = document.getElementById('text-length');

            const prob = data.probability * 100;
            percentage.textContent = prob.toFixed(1) + '%';

            // Set result text and color
            if (data.probability > 0.5) {
                resultText.textContent = data.result;
                resultText.className = 'result-text ai';
                progressFill.style.background = `linear-gradient(90deg, #f39c12 0%, #e74c3c ${prob}%)`;
            } else {
                resultText.textContent = data.result;
                resultText.className = 'result-text human';
                progressFill.style.background = `linear-gradient(90deg, #27ae60 0%, #f39c12 ${prob}%)`;
            }

            progressFill.style.width = prob + '%';
            textLength.textContent = data.text_length + ' 字符';

            resultCard.classList.add('show');
        }

        function showError(message) {
            const errorMsg = document.getElementById('error-message');
            errorMsg.textContent = message;
            errorMsg.classList.add('show');
        }

        // Allow Ctrl+Enter to submit
        document.getElementById('text-input').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                detectText();
            }
        });
    </script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add templates/index.html
git commit -m "feat: add frontend HTML with Clean Modern design"
```

---

### Task 3: Test the Application

**Files:**
- Test: Run the application

- [ ] **Step 1: Install dependencies**

```bash
pip install -r requirements_web.txt
```

- [ ] **Step 2: Start Flask server**

```bash
python app.py
```

Expected: Model loads, server starts on port 5000

- [ ] **Step 3: Test in browser**

Open http://localhost:5000

- [ ] **Step 4: Test detection with sample text**

Enter some Chinese text and click 检测

Expected: Returns probability score

- [ ] **Step 5: Commit**

```bash
git add requirements_web.txt
git commit -m "chore: add web app dependencies and finalize"
```

---

## Summary

After completing these tasks, you will have:
1. ✅ Flask backend (`app.py`) serving the AIGC model
2. ✅ Clean Modern frontend (`templates/index.html`)
3. ✅ Working detection at http://localhost:5000
