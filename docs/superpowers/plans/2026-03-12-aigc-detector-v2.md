# AIGC 文本检测器 v2 - 实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan.

**Goal:** 实现文献阅读器风格的 AIGC 检测界面，支持多级别分块、荧光笔高亮、点击编辑功能。

**Architecture:** Flask 后端提供新接口，前端使用双栏布局实现阅读器风格。

**Tech Stack:** Flask, python-docx, Vanilla JS, HTML/CSS

---

## 文件结构

```
AIGC_detector_zhv3/
├── app.py                      # Flask 应用 (更新)
├── templates/
│   └── index.html              # 前端界面 (全新)
└── ...
```

---

## 实现任务

### Task 1: 修复 Word 解析编码问题

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 修复 upload 函数中的编码处理**

更新 `/api/upload` 函数，添加多种编码尝试：

```python
def read_file_content(file, filename):
    """尝试多种编码读取文件"""
    # 尝试的编码顺序
    encodings = ['utf-8', 'gbk', 'gb18030', 'utf-16', 'latin1']

    if filename.endswith('.txt'):
        content = file.read()
        for encoding in encodings:
            try:
                return content.decode(encoding)
            except (UnicodeDecodeError, AttributeError):
                continue
        # 最后尝试不带编码
        return content.decode('utf-8', errors='ignore')

    elif filename.endswith('.docx'):
        # 尝试使用不同方式读取 docx
        try:
            # 方式1: 直接读取
            doc = Document(file)
            return '\n'.join([para.text for para in doc.paragraphs])
        except:
            # 方式2: 从BytesIO读取
            from io import BytesIO
            file.seek(0)
            doc = Document(BytesIO(file.read()))
            return '\n'.join([para.text for para in doc.paragraphs])

    raise ValueError("不支持的文件格式")
```

- [ ] **Step 2: 更新 /api/upload 接口**

```python
@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '请选择文件'}), 400

        file = request.files['file']
        filename = file.filename.lower()

        if not (filename.endswith('.txt') or filename.endswith('.docx')):
            return jsonify({'success': False, 'error': '不支持的文件格式'}), 400

        text = read_file_content(file, filename)

        return jsonify({
            'success': True,
            'text': text,
            'filename': file.filename
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
```

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "fix: improve Word/TXT file encoding handling"
```

---

### Task 2: 添加多级别检测 API

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 添加文本分块函数**

```python
def split_by_paragraphs(text):
    """按段落分块"""
    paragraphs = text.replace('\r\n', '\n').split('\n')
    return [p.strip() for p in paragraphs if p.strip()]

def split_by_sentences(text):
    """按句子分块"""
    import re
    # 按句子结束符分割：。！？.
    sentences = re.split(r'([。！？.!?])', text)
    chunks = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i]
        punct = sentences[i + 1] if i + 1 < len(sentences) else ""
        if (sentence + punct).strip():
            chunks.append(sentence + punct)
    if sentences and sentences[-1].strip():
        chunks.append(sentences[-1])
    return [c.strip() for c in chunks if c.strip()]

def detect_chunk(text):
    """检测单个块的 AIGC 概率"""
    inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        return probs[0][0].item()
```

- [ ] **Step 2: 添加 /api/detect-full 接口**

```python
@app.route('/api/detect-full', methods=['POST'])
def detect_full():
    try:
        data = request.get_json()
        text = data.get('text', '')
        mode = data.get('mode', 'paragraph')  # paragraph 或 sentence

        if not text or not text.strip():
            return jsonify({'success': False, 'error': '请输入文本'}), 400

        # 分块
        if mode == 'sentence':
            chunks = split_by_sentences(text)
        else:
            chunks = split_by_paragraphs(text)

        # 检测每个块
        results = []
        total_chars = 0
        weighted_sum = 0

        for i, chunk in enumerate(chunks):
            prob = detect_chunk(chunk)
            chunk_len = len(chunk)
            total_chars += chunk_len
            weighted_sum += prob * chunk_len

            results.append({
                'index': i,
                'text': chunk,
                'probability': round(prob, 4),
                'text_length': chunk_len
            })

        # 计算整体概率
        overall = weighted_sum / total_chars if total_chars > 0 else 0

        return jsonify({
            'success': True,
            'overall_probability': round(overall, 4),
            'mode': mode,
            'chunks': results,
            'text_length': len(text)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
```

- [ ] **Step 3: 添加 /api/detect-chunk 接口（单块检测）**

```python
@app.route('/api/detect-chunk', methods=['POST'])
def detect_chunk_api():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text or not text.strip():
            return jsonify({'success': False, 'error': '请输入文本'}), 400

        prob = detect_chunk(text)

        return jsonify({
            'success': True,
            'probability': round(prob, 4)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
```

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add multi-level detection APIs"
```

---

### Task 3: 创建文献阅读器风格前端

**Files:**
- Modify: `templates/index.html`

- [ ] **Step 1: 编写完整的 HTML/CSS/JS**

创建双栏布局：

```html
<!-- CSS -->
<style>
  :root {
    --bg: #ffffff;
    --bg-secondary: #f7f7f5;
    --text: #1a1a1a;
    --primary: #d97706;
    --border: #e5e5e5;
    --risk-high: #ef4444;
    --risk-high-bg: #fee2e2;
    --risk-medium: #f59e0b;
    --risk-medium-bg: #fef3c7;
    --risk-low: #22c55e;
    --risk-low-bg: #dcfce7;
  }

  .container {
    display: flex;
    height: calc(100vh - 144px);
  }

  .reader-panel {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
    border-right: 1px solid var(--border);
  }

  .sidebar {
    width: 320px;
    padding: 20px;
    background: var(--bg-secondary);
    overflow-y: auto;
  }

  .chunk-item {
    padding: 12px;
    margin-bottom: 8px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .chunk-item:hover {
    transform: translateX(4px);
  }

  .chunk-item.selected {
    outline: 2px solid var(--primary);
  }

  .chunk-item.high { background: var(--risk-high-bg); }
  .chunk-item.medium { background: var(--risk-medium-bg); }
  .chunk-item.low { background: var(--risk-low-bg); }

  .mode-switch {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
  }

  .mode-btn {
    padding: 8px 16px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: white;
    cursor: pointer;
  }

  .mode-btn.active {
    background: var(--primary);
    color: white;
    border-color: var(--primary);
  }

  .editor-area {
    margin-top: 20px;
  }

  .editor-area textarea {
    width: 100%;
    min-height: 150px;
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 14px;
    resize: vertical;
  }

  .overall-stats {
    position: sticky;
    bottom: 0;
    background: white;
    padding: 16px;
    border-top: 1px solid var(--border);
  }
</style>
```

- [ ] **Step 2: JavaScript 交互逻辑**

```javascript
// 主要函数
let currentMode = 'paragraph';
let chunksData = [];
let selectedChunk = null;

function loadFile() { /* 文件上传 */ }

function runDetection() {
  const text = document.getElementById('text-input').value;
  fetch('/api/detect-full', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text, mode: currentMode})
  })
  .then(r => r.json())
  .then(data => {
    chunksData = data.chunks;
    renderChunks();
    updateOverall(data.overall_probability);
  });
}

function renderChunks() {
  // 渲染每个块，带荧光笔颜色
}

function selectChunk(index) {
  selectedChunk = chunksData[index];
  renderEditor();
}

function renderEditor() {
  // 右侧操作栏显示选中块
  // 带编辑功能，编辑后自动重新检测
}

// 防抖自动检测
let debounceTimer;
function onEditorInput() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    const text = document.getElementById('edit-text').value;
    fetch('/api/detect-chunk', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    })
    .then(r => r.json())
    .then(data => {
      // 更新当前块的概率
      updateSelectedChunk(data.probability);
    });
  }, 500);
}
```

- [ ] **Step 3: Commit**

```bash
git add templates/index.html
git commit -m "feat: add reader-style frontend with highlight and editor"
```

---

### Task 4: 测试新功能

**Files:**
- Test: 运行测试

- [ ] **Step 1: 启动服务器测试**

```bash
python app.py
```

- [ ] **Step 2: 测试 Word 文件上传**

- [ ] **Step 3: 测试段落/句子级别切换**

- [ ] **Step 4: 测试点击选中功能**

- [ ] **Step 5: 测试编辑自动检测**

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "test: verify v2 features work correctly"
```

---

## 验收标准

1. ✅ Word 文件导入无乱码
2. ✅ 段落级/句子级切换正常
3. ✅ 荧光笔高亮显示正确（红/黄/绿）
4. ✅ 点击段落/句子在操作栏显示
5. ✅ 编辑文本后自动重新检测
6. ✅ 整体 AIGC 率实时更新
