# AIGC 文本检测器增强版 - 实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个增强版 AIGC 文本检测器，支持文本粘贴、文件导入（TXT/Word）、整体和分块检测功能，采用 Claude Code 官网风格界面。

**Architecture:** Flask 后端提供 REST API，前端使用 Vanilla JS + CSS。复用现有的 PyTorch 模型进行推理。

**Tech Stack:** Flask, Flask-CORS, python-docx, Transformers, PyTorch, HTML/CSS/JS

---

## 文件结构

```
AIGC_detector_zhv3/
├── app.py                      # Flask 应用 (更新 - 添加新接口)
├── requirements_web.txt        # Python 依赖 (更新 - 添加 python-docx)
├── templates/
│   └── index.html              # 前端界面 (全新设计)
└── ...
```

---

## 实现任务

### Task 1: 更新 Python 依赖

**Files:**
- Modify: `requirements_web.txt`

- [ ] **Step 1: 添加 python-docx 依赖**

```txt
flask>=2.0
flask-cors>=3.0
torch>=1.10
transformers>=4.27
python-docx>=0.8
```

- [ ] **Step 2: 安装新依赖**

```bash
pip install python-docx
```

- [ ] **Step 3: Commit**

```bash
git add requirements_web.txt
git commit -m "chore: add python-docx dependency"
```

---

### Task 2: 更新 Flask 后端 (添加新接口)

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 添加导入和文本分块函数**

在文件开头添加导入:
```python
import re
from docx import Document
```

添加文本分块函数:
```python
def split_text_into_chunks(text, chunk_size):
    """将文本分割成块，保证句子完整"""
    # 先按句子分割
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
            # 如果单个句子超过 chunk_size，则强制分割
            if len(combined) > chunk_size:
                # 按字符强制分割
                for j in range(0, len(combined), chunk_size):
                    chunks.append(combined[j:j+chunk_size])
                current_chunk = ""
            else:
                current_chunk = combined

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
```

- [ ] **Step 2: 添加文件上传接口**

在 `/api/detect` 接口后添加:
```python
@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': '请选择文件'
            }), 400

        file = request.files['file']
        filename = file.filename.lower()

        if filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        elif filename.endswith('.docx'):
            doc = Document(file)
            text = '\n'.join([para.text for para in doc.paragraphs])
        else:
            return jsonify({
                'success': False,
                'error': '不支持的文件格式，请使用 .txt 或 .docx 文件'
            }), 400

        return jsonify({
            'success': True,
            'text': text,
            'filename': file.filename
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

- [ ] **Step 3: 添加分块检测接口**

在 `/api/upload` 接口后添加:
```python
@app.route('/api/detect-chunks', methods=['POST'])
def detect_chunks():
    try:
        data = request.get_json()
        text = data.get('text', '')
        chunk_size = data.get('chunk_size', 200)  # 默认 200 字

        if not text or not text.strip():
            return jsonify({
                'success': False,
                'error': '请输入要检测的文本'
            }), 400

        # 分割文本
        chunks = split_text_into_chunks(text, chunk_size)

        # 检测每个块
        results = []
        for i, chunk in enumerate(chunks):
            inputs = tokenizer(
                chunk,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

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

        # 计算整体概率 (加权平均)
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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add chunk detection and file upload APIs"
```

---

### Task 3: 创建前端界面

**Files:**
- Modify: `templates/index.html`

- [ ] **Step 1: 编写完整的 HTML/CSS/JS**

创建 `templates/index.html`，包含:
- Claude Code 官网风格 CSS
- 文件上传区域
- 文本编辑区域
- 块大小选择器
- 整体结果展示
- 分块结果列表

完整的 CSS 变量:
```css
:root {
  --bg: #ffffff;
  --bg-secondary: #f7f7f5;
  --text: #1a1a1a;
  --text-secondary: #6e6e6e;
  --primary: #d97706;
  --primary-hover: #b45309;
  --primary-text: #ffffff;
  --border: #e5e5e5;
  --muted: #737373;
  --container: 1200px;
  --font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-display: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --risk-high: #ef4444;
  --risk-medium: #f59e0b;
  --risk-low: #22c55e;
}
```

完整的 JavaScript 功能:
- `handleFileUpload()` - 处理文件上传
- `detectText()` - 检测整体文本
- `detectChunks()` - 分块检测
- `renderResults()` - 渲染结果
- `getRiskLevel()` - 获取风险等级

- [ ] **Step 2: Commit**

```bash
git add templates/index.html
git commit -m "feat: add new frontend with Claude Code style"
```

---

### Task 4: 测试新功能

**Files:**
- Test: 运行应用并测试

- [ ] **Step 1: 确保服务器运行**

```bash
# 如果服务器没运行，启动它
python app.py
```

- [ ] **Step 2: 测试文件上传 API**

```bash
# 创建测试文件
echo "测试文本内容" > test.txt

# 测试上传
curl -X POST -F "file=@test.txt" http://127.0.0.1:5000/api/upload
```

- [ ] **Step 3: 测试分块检测 API**

```bash
curl -X POST http://127.0.0.1:5000/api/detect-chunks \
  -H "Content-Type: application/json" \
  -d '{"text": "这是一个很长的文本。我们需要把它分割成多个小块。每一块都会进行独立的AI检测。这样用户可以看到哪些部分是AI生成的。", "chunk_size": 50}'
```

- [ ] **Step 4: 在浏览器测试**

1. 打开 http://localhost:5000
2. 粘贴或导入文本
3. 选择块大小
4. 点击检测按钮
5. 查看结果

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "test: verify new features work correctly"
```

---

## 总结

完成这些任务后，你将拥有:
1. ✅ 支持 TXT/Word 文件导入
2. ✅ 整体文本 AIGC 检测
3. ✅ 分块检测功能
4. ✅ 可选块大小 (100/200/500/1000字)
5. ✅ Claude Code 官网风格界面
6. ✅ 风险可视化展示
