# Plaintext/Word Mode Toggle Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add plaintext mode as default entry point with toggle to Word mode, showing warning modal when switching to Word mode.

**Architecture:** Add mode state management (`editorMode: 'plaintext' | 'word'`) with conditional rendering of either a textarea (plaintext) or SuperDocEditor (Word). Word mode triggers a warning modal about broken jump functionality.

**Tech Stack:** React (frontend), TypeScript, CSS custom properties

---

## File Structure

```
frontend/src/
├── App.tsx              # Main component - add state, mode toggle, modal, plaintext editor
└── index.css            # Add styles for modal and plaintext editor
```

---

## Chunk 1: Add State and Icon

**Files:**
- Modify: `frontend/src/App.tsx:1-70` (add icon and state)

- [ ] **Step 1: Add IconPlainText SVG icon after IconMoon (line 58)**

```tsx
const IconPlainText = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="17" y1="10" x2="3" y2="10" />
    <line x1="21" y1="6" x2="3" y2="6" />
    <line x1="21" y1="14" x2="3" y2="14" />
    <line x1="17" y1="18" x2="3" y2="18" />
  </svg>
)
```

- [ ] **Step 2: Add new state variables after line 165 (after theme state)**

```tsx
const [editorMode, setEditorMode] = useState<'plaintext' | 'word'>(() => {
  const saved = localStorage.getItem('editorMode')
  return (saved === 'word' ? 'word' : 'plaintext')
})
const [showWordWarning, setShowWordWarning] = useState(false)
const [plainTextContent, setPlainTextContent] = useState('')
```

- [ ] **Step 3: Add useEffect to persist editorMode to localStorage (after theme useEffect)**

```tsx
useEffect(() => {
  localStorage.setItem('editorMode', editorMode)
}, [editorMode])
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: add editorMode state, IconPlainText, and localStorage persistence"
```

---

## Chunk 2: Add Mode Toggle Buttons to Sidebar

**Files:**
- Modify: `frontend/src/App.tsx:540-570` (add mode toggle buttons)

- [ ] **Step 1: Add mode toggle section between Export dropdown and Theme toggle (after line 561, before sidebar-divider)**

```tsx
<div className="sidebar-divider" />

{/* 模式切换 - 位于导出和主题之间 */}
<div className="mode-toggle">
  <button
    className={`mode-btn ${editorMode === 'plaintext' ? 'active' : ''}`}
    onClick={() => {
      if (editorMode !== 'plaintext') {
        setEditorMode('plaintext')
      }
    }}
    aria-label="纯文本模式"
    title="纯文本模式"
  >
    <IconPlainText />
    <span>纯文本</span>
  </button>
  <button
    className={`mode-btn ${editorMode === 'word' ? 'active' : ''}`}
    onClick={() => {
      if (editorMode !== 'word') {
        setShowWordWarning(true)
      }
    }}
    aria-label="Word模式"
    title="Word模式"
  >
    <IconFile />
    <span>Word</span>
  </button>
</div>

<div className="sidebar-divider" />

{/* 主题切换 */}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: add mode toggle buttons to sidebar"
```

---

## Chunk 3: Add WordWarningModal Component

**Files:**
- Modify: `frontend/src/App.tsx` (add modal component and state)

- [ ] **Step 1: Add WordWarningModal component before App function (after getAILevel function, around line 150)**

```tsx
const WordWarningModal = ({
  onConfirm,
  onCancel
}: {
  onConfirm: () => void
  onCancel: () => void
}) => {
  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <span className="modal-icon">⚠️</span>
          <span className="modal-title">提示</span>
        </div>
        <div className="modal-body">
          <p className="modal-message">Word 模式跳转功能正在修复中</p>
          <p className="modal-detail">当前问题：点击右侧分块无法跳转到左侧 Word 编辑器的对应位置</p>
        </div>
        <div className="modal-footer">
          <button className="btn btn-primary" onClick={onConfirm}>
            仍然切换到 Word 模式（编辑功能可用）
          </button>
          <button className="btn btn-secondary" onClick={onCancel}>
            留在纯文本模式
          </button>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Add modal rendering in App component return (after sidebar, before main-area)**

```tsx
{/* 左侧工具栏 */}
<aside className="sidebar" ...>

{/* Word模式警告弹窗 */}
{showWordWarning && (
  <WordWarningModal
    onConfirm={() => {
      setEditorMode('word')
      setShowWordWarning(false)
    }}
    onCancel={() => setShowWordWarning(false)}
  />
)}

{/* 主区域 */}
<div className="main-area">
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: add WordWarningModal component"
```

---

## Chunk 4: Add PlainTextEditor UI

**Files:**
- Modify: `frontend/src/App.tsx` (modify editor panel rendering)

- [ ] **Step 1: Add PlainTextEditor component after WordWarningModal**

```tsx
const PlainTextEditor = ({
  value,
  onChange,
  onDetect
}: {
  value: string
  onChange: (text: string) => void
  onDetect: () => void
}) => {
  return (
    <div className="plaintext-editor">
      <textarea
        className="plaintext-textarea"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="请在此输入或粘贴需要检测的文本..."
        spellCheck={false}
      />
      <div className="plaintext-actions">
        <button
          className="btn btn-secondary"
          onClick={() => {
            const totalChars = value.length
            const chunks = Math.ceil(totalChars / 500)
            alert(`总计 ${totalChars} 字，预计分为 ${chunks} 个段落进行检测`)
          }}
        >
          字数统计
        </button>
        <button
          className="btn btn-primary"
          onClick={onDetect}
          disabled={!value.trim()}
        >
          开始检测
        </button>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Modify the editor panel content to render based on editorMode (around line 606)**

Replace the current panel-content:
```tsx
<div className="panel-content">
  {editorMode === 'plaintext' ? (
    <PlainTextEditor
      value={plainTextContent}
      onChange={setPlainTextContent}
      onDetect={() => {
        if (!plainTextContent.trim()) {
          setError('请先输入文本内容')
          return
        }
        setEditorPlainText(plainTextContent)
        setIsDetecting(true)
        setError(null)
        detectAPI(plainTextContent, chunkSize)
          .then(setDetectionResult)
          .catch((err) => setError(err instanceof Error ? err.message : '检测失败'))
          .finally(() => setIsDetecting(false))
      }}
    />
  ) : docFile ? (
    <>
      <SuperDocEditor ... />
      {isDetecting && <div className="loading-overlay"><div className="spinner" /></div>}
    </>
  ) : (
    <div className="empty-state">
      <div className="empty-content">
        <h3>暂无文档</h3>
        <p>点击左上角按钮打开 .docx 文件</p>
        <button className="btn btn-primary" onClick={() => fileInputRef.current?.click()}>
          打开文档
        </button>
      </div>
    </div>
  )}
</div>
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: add PlainTextEditor component and mode-based rendering"
```

---

## Chunk 5: Add Styles for Modal and Plaintext Editor

**Files:**
- Modify: `frontend/src/index.css`

- [ ] **Step 1: Add modal styles (at end of file)**

```css
/* Word Warning Modal */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: var(--theme-surface);
  border-radius: 8px;
  padding: 24px;
  max-width: 420px;
  width: 90%;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}

.modal-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
}

.modal-icon {
  font-size: 24px;
}

.modal-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--theme-text);
}

.modal-body {
  margin-bottom: 20px;
}

.modal-message {
  font-size: 15px;
  color: var(--theme-text);
  margin-bottom: 8px;
}

.modal-detail {
  font-size: 13px;
  color: var(--theme-text-muted);
}

.modal-footer {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.modal-footer .btn {
  width: 100%;
  padding: 12px 16px;
  font-size: 14px;
}
```

- [ ] **Step 2: Add mode toggle styles**

```css
/* Mode Toggle */
.mode-toggle {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 4px;
}

.mode-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  border: none;
  background: transparent;
  color: var(--theme-text-muted);
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s ease;
}

.mode-btn:hover {
  background: var(--theme-hover);
  color: var(--theme-text);
}

.mode-btn.active {
  background: var(--theme-primary);
  color: white;
}

.mode-btn svg {
  width: 18px;
  height: 18px;
}
```

- [ ] **Step 3: Add plaintext editor styles**

```css
/* Plaintext Editor */
.plaintext-editor {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 16px;
  gap: 12px;
}

.plaintext-textarea {
  flex: 1;
  width: 100%;
  padding: 16px;
  border: 1px solid var(--theme-border);
  border-radius: 8px;
  background: var(--theme-background);
  color: var(--theme-text);
  font-family: var(--font-display);
  font-size: 14px;
  line-height: 1.8;
  resize: none;
}

.plaintext-textarea:focus {
  outline: none;
  border-color: var(--theme-primary);
}

.plaintext-textarea::placeholder {
  color: var(--theme-text-muted);
}

.plaintext-actions {
  display: flex;
  gap: 10px;
  justify-content: flex-end;
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/index.css
git commit -m "style: add modal, mode toggle, and plaintext editor styles"
```

---

## Chunk 6: Build and Test

**Files:**
- Modify: `frontend/` (build output)

- [ ] **Step 1: Build frontend**

```bash
cd frontend && npm run build
```

Expected: Build completes without errors, output in `frontend/dist/`

- [ ] **Step 2: Verify the implementation**
- Check that Flask is running on port 5000
- Access http://localhost:5000
- Verify:
  1. Default mode is plaintext (textarea visible)
  2. Clicking "Word" button shows warning modal
  3. Clicking "仍然切换" switches to Word mode
  4. Clicking "留在纯文本模式" closes modal
  5. Mode selection persists after page reload

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: implement plaintext/word mode toggle with warning modal"
```

---

## Verification Checklist

- [ ] Default page shows plaintext textarea
- [ ] Clicking "Word" button shows warning modal
- [ ] Modal "仍然切换" button switches to Word mode
- [ ] Modal "留在纯文本模式" closes modal without switching
- [ ] Word mode shows SuperDoc editor when file is loaded
- [ ] Mode selection persists after page reload
- [ ] No console errors
