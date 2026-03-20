import { useState, useRef, useCallback, useEffect, useMemo, memo } from 'react'
import { SuperDocEditor } from '@superdoc-dev/react'
import type { SuperDocRef } from '@superdoc-dev/react'
import '@superdoc-dev/react/style.css'

// SVG Icons
const IconFile = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14,2 14,8 20,8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
    <polyline points="10,9 9,9 8,9" />
  </svg>
)

const IconSearch = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
)

const IconChart = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="20" x2="18" y2="10" />
    <line x1="12" y1="20" x2="12" y2="4" />
    <line x1="6" y1="20" x2="6" y2="14" />
  </svg>
)

const IconDownload = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="7,10 12,15 17,10" />
    <line x1="12" y1="15" x2="12" y2="3" />
  </svg>
)

const IconSun = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="5" />
    <line x1="12" y1="1" x2="12" y2="3" />
    <line x1="12" y1="21" x2="12" y2="23" />
    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
    <line x1="1" y1="12" x2="3" y2="12" />
    <line x1="21" y1="12" x2="23" y2="12" />
    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
  </svg>
)

const IconMoon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
  </svg>
)

const IconPlainText = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="17" y1="10" x2="3" y2="10" />
    <line x1="21" y1="6" x2="3" y2="6" />
    <line x1="21" y1="14" x2="3" y2="14" />
    <line x1="17" y1="18" x2="3" y2="18" />
  </svg>
)

// 类型定义
interface ChunkDetection {
  text: string
  probability: number
  index: number
  text_length: number
  start_pos: number
  end_pos?: number
}

interface DetectionResult {
  chunks: ChunkDetection[]
  overall_probability: number
  text_length: number
  mode: string
}

interface SystemInfo {
  cpu: { percent: number; count: number }
  memory: { used_gb: number; total_gb: number; percent: number }
  gpu: { name: string; memory_allocated_gb: number; memory_reserved_gb: number; memory_total_gb: number } | null
  model: { loaded: boolean }
  system: { platform: string; python_version: string }
}

// API函数
const detectAPI = async (text: string, chunkSize: string = 'original'): Promise<DetectionResult> => {
  const response = await fetch('/api/detect-full', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, mode: 'paragraph', chunk_size: chunkSize })
  })
  const data = await response.json()
  if (!data.success) throw new Error(data.error || '检测失败')
  return data
}

// 单块检测API
const detectChunkAPI = async (text: string): Promise<{ probability: number }> => {
  const response = await fetch('/api/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  })
  const data = await response.json()
  if (!data.success) throw new Error(data.error || '检测失败')
  return { probability: data.probability }
}

// 系统信息API
const getSystemInfoAPI = async (): Promise<SystemInfo> => {
  const response = await fetch('/api/system-info')
  const data = await response.json()
  if (!data.success) throw new Error(data.error || '获取系统信息失败')
  return data
}

// 工具栏按钮组件
const ToolbarButton = ({
  icon,
  label,
  active,
  onClick,
  disabled,
  showDropdown,
  dropdownContent,
  ariaLabel
}: {
  icon: React.ReactNode
  label: string
  active?: boolean
  onClick?: () => void
  disabled?: boolean
  showDropdown?: boolean
  dropdownContent?: React.ReactNode
  ariaLabel?: string
}) => {
  const [show, setShow] = useState(false)

  return (
    <div className="dropdown">
      <button
        className={`sidebar-btn tooltip ${active ? 'active' : ''}`}
        data-tooltip={label}
        aria-label={ariaLabel || label}
        onClick={onClick}
        disabled={disabled}
        onMouseEnter={() => showDropdown && setShow(true)}
        onMouseLeave={() => setShow(false)}
      >
        {icon}
      </button>
      {showDropdown && show && (
        <div className="dropdown-menu" role="menu" onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
          {dropdownContent}
        </div>
      )}
    </div>
  )
}

const getAILevel = (probability: number): 'high' | 'medium' | 'low' | 'human' => {
  if (probability > 0.7) return 'high'
  if (probability > 0.4) return 'medium'
  if (probability > 0.2) return 'low'
  return 'human'
}

// Memoized chunk component to prevent unnecessary re-renders
interface TextChunkProps {
  chunk: ChunkDetection
  index: number
  isJumping: boolean
  isSelected: boolean
  isEditing: boolean
  isDetecting: boolean
  onSelect: (index: number) => void
  onEdit: (index: number) => void
  onCancelEdit: () => void
  onReDetect: (index: number, text: string) => void
  onJumpingChange: (index: number | null) => void
}

const TextChunk = memo(function TextChunk({
  chunk,
  index,
  isJumping,
  isSelected,
  isEditing,
  isDetecting,
  onSelect,
  onEdit,
  onCancelEdit,
  onReDetect,
  onJumpingChange
}: TextChunkProps) {
  const aiLevel = getAILevel(chunk.probability)

  return (
    <div
      key={index}
      ref={(el) => {
        if (isJumping && el) {
          setTimeout(() => {
            el.scrollIntoView({ behavior: 'smooth', block: 'center' })
            onJumpingChange(null)
          }, 50)
        }
      }}
      className={`text-chunk ${aiLevel} ${isSelected ? 'selected' : ''} ${isJumping ? 'jump-to' : ''}`}
      onClick={() => {
        if (!isEditing) {
          onSelect(index)
          onJumpingChange(index)
        }
      }}
    >
      {/* 元数据行 - 始终显示 */}
      <div className="chunk-meta-row">
        <span className={`ai-badge ${aiLevel}`}>
          {Math.round(chunk.probability * 100)}%
        </span>
        <span className="chunk-meta">{chunk.text_length} 字</span>
        {!isEditing && (
          <button
            className="btn btn-secondary chunk-edit-btn"
            onClick={(e) => {
              e.stopPropagation()
              onEdit(index)
            }}
          >
            编辑
          </button>
        )}
      </div>

      {isEditing ? (
        <div onClick={(e) => e.stopPropagation()}>
          <textarea
            className="chunk-editor"
            defaultValue={chunk.text}
            placeholder="在此修改文本..."
          />
          <div className="chunk-actions">
            <button
              className="btn btn-secondary"
              onClick={(e) => {
                e.stopPropagation()
                onCancelEdit()
              }}
            >
              取消
            </button>
            <button
              className="btn btn-primary"
              disabled={isDetecting}
              onClick={(e) => {
                e.stopPropagation()
                const textarea = document.querySelector(`.chunk-editor`) as HTMLTextAreaElement
                if (textarea) {
                  onReDetect(index, textarea.value)
                }
              }}
            >
              {isDetecting ? '检测中...' : '重新检测'}
            </button>
          </div>
        </div>
      ) : (
        <div className="chunk-text">
          {chunk.text}
        </div>
      )}
    </div>
  )
})

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

const PlainTextEditor = ({
  value,
  onChange
}: {
  value: string
  onChange: (text: string) => void
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
    </div>
  )
}

function App() {
  const [docFile, setDocFile] = useState<File | null>(null)
  const [mode] = useState<'editing' | 'viewing' | 'suggesting'>('editing')
  const [isReady, setIsReady] = useState(false)
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null)
  const [isDetecting, setIsDetecting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [chunkSize, setChunkSize] = useState<string>('original')
  const [selectedChunk, setSelectedChunk] = useState<number | null>(null)
  const [jumpingChunk, setJumpingChunk] = useState<number | null>(null)
  const [editorPlainText, setEditorPlainText] = useState<string>('')
  const [detectingChunk, setDetectingChunk] = useState<number | null>(null)
  const [editingChunk, setEditingChunk] = useState<number | null>(null)
  const [theme, setTheme] = useState<'light' | 'dark'>('light')
  const [editorMode, setEditorMode] = useState<'plaintext' | 'word'>(() => {
    const saved = localStorage.getItem('editorMode')
    return (saved === 'word' ? 'word' : 'plaintext')
  })
  const [showWordWarning, setShowWordWarning] = useState(false)
  const [plainTextContent, setPlainTextContent] = useState('')
  const [splitRatio, setSplitRatio] = useState(0.5)
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null)

  const editorRef = useRef<SuperDocRef>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // 主题切换
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') as 'light' | 'dark' | null
    if (savedTheme) {
      setTheme(savedTheme)
      document.documentElement.setAttribute('data-theme', savedTheme)
    } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      setTheme('dark')
      document.documentElement.setAttribute('data-theme', 'dark')
    }
  }, [])

  useEffect(() => {
    localStorage.setItem('editorMode', editorMode)
  }, [editorMode])

  // 纯文本模式下直接设置 isReady 为 true
  useEffect(() => {
    if (editorMode === 'plaintext') {
      setIsReady(true)
    }
  }, [editorMode])

  // 定期获取系统信息
  useEffect(() => {
    const fetchSystemInfo = async () => {
      try {
        const info = await getSystemInfoAPI()
        setSystemInfo(info)
      } catch (err) {
        console.warn('获取系统信息失败:', err)
      }
    }

    // 立即获取一次
    fetchSystemInfo()

    // 每5秒更新一次
    const interval = setInterval(fetchSystemInfo, 5000)
    return () => clearInterval(interval)
  }, [])

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    setTheme(newTheme)
    localStorage.setItem('theme', newTheme)
    document.documentElement.setAttribute('data-theme', newTheme)
  }

  const scrollToChunk = useCallback((index: number) => {
    const instance = editorRef.current?.getInstance()
    if (!instance || !detectionResult) return

    const chunk = detectionResult.chunks[index]
    if (!chunk) return

    const targetText = chunk.text
    if (!targetText) return

    // 方法1: 尝试使用 SuperDoc 的搜索 API 直接搜索并跳转
    try {
      if (typeof instance.search === 'function') {
        const results = instance.search(targetText)
        if (results && results.length > 0) {
          const match = results[0]
          if (typeof instance.goToSearchResult === 'function') {
            instance.goToSearchResult(match)
            return
          }
        }
      }
    } catch (e) {
      console.warn('SuperDoc search navigation failed:', e)
    }

    // 方法2: 尝试使用 scrollToPosition (ProseMirror 位置)
    try {
      const activeEditor = (instance as any)?.activeEditor
      if (activeEditor) {
        // 获取编辑器纯文本
        let editorText = ''
        if (typeof instance.getText === 'function') {
          editorText = instance.getText()
        }

        // 使用后端返回的 start_pos 找到目标位置
        const targetStartPos = chunk.start_pos || 0
        let targetPosInEditor = -1

        // 验证 start_pos 是否有效
        if (targetStartPos >= 0 && targetStartPos < editorText.length) {
          const editorChunk = editorText.substring(targetStartPos, targetStartPos + targetText.length)
          if (editorChunk === targetText) {
            targetPosInEditor = targetStartPos
          }
        }

        // 如果后端位置不匹配，使用模糊搜索
        if (targetPosInEditor === -1) {
          const searchRadius = 50
          for (let i = Math.max(0, targetStartPos - searchRadius); i <= Math.min(editorText.length - targetText.length, targetStartPos + searchRadius); i++) {
            if (editorText.substring(i, i + targetText.length) === targetText) {
              targetPosInEditor = i
              break
            }
          }
        }

        // 尝试滚动
        if (targetPosInEditor >= 0) {
          if (activeEditor.scrollToPositionAsync) {
            activeEditor.scrollToPositionAsync(targetPosInEditor, { block: 'center', behavior: 'smooth' })
            return
          }
          if (activeEditor.scrollToPosition) {
            activeEditor.scrollToPosition(targetPosInEditor, { block: 'center', behavior: 'smooth' })
            return
          }
        }
      }
    } catch (e) {
      console.warn('scrollToPosition method failed:', e)
    }

    // 最终 fallback: 聚焦编辑器
    try {
      if (typeof instance.focus === 'function') {
        instance.focus()
      }
    } catch (e) {
      console.warn('Editor focus failed:', e)
    }

  }, [detectionResult])

  const handleDetect = useCallback(async () => {
    setIsDetecting(true)
    setError(null)
    try {
      let plainText = ''

      // 纯文本模式：直接使用 plainTextContent
      if (editorMode === 'plaintext') {
        plainText = plainTextContent
      } else {
        // Word 模式：从编辑器实例获取文本
        const instance = editorRef.current?.getInstance()
        if (!instance) {
          setError('编辑器未加载')
          return
        }
        if (typeof instance.getText === 'function') {
          plainText = instance.getText()
        }
        if (!plainText || plainText.length < 10) {
          const editorText = instance.getHTML()
          const div = window.document.createElement('div')
          div.innerHTML = editorText
          plainText = div.textContent || div.innerText || ''
        }
      }

      if (!plainText.trim()) {
        setError('请先输入或导入文本内容')
        return
      }

      setEditorPlainText(plainText)
      const result = await detectAPI(plainText, chunkSize)
      setDetectionResult(result)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '检测失败，请稍后重试'
      setError(errorMessage)
    } finally {
      setIsDetecting(false)
    }
  }, [chunkSize, editorMode, plainTextContent])

  // 重新检测单个块
  const handleChunkDetect = useCallback(async (index: number, newText: string) => {
    if (!detectionResult) return

    setDetectingChunk(index)
    try {
      const result = await detectChunkAPI(newText)

      // 更新该块的概率
      const newChunks = [...detectionResult.chunks]
      newChunks[index] = {
        ...newChunks[index],
        text: newText,
        probability: result.probability,
        text_length: newText.length
      }

      // 重新计算整体概率
      const totalProb = newChunks.reduce((sum, c) => sum + c.probability, 0)
      const overallProb = totalProb / newChunks.length

      setDetectionResult({
        ...detectionResult,
        chunks: newChunks,
        overall_probability: overallProb,
        text_length: newChunks.reduce((sum, c) => sum + c.text_length, 0)
      })
    } catch (err) {
      console.error('单块检测失败:', err)
    } finally {
      setDetectingChunk(null)
      setEditingChunk(null)
    }
  }, [detectionResult])

  // 合并编辑后的文本到原文
  const handleMergeToOriginal = useCallback(() => {
    if (!detectionResult) return

    // 按顺序拼接所有chunk的文本，用换行符分隔保留原有段落结构
    const mergedText = detectionResult.chunks.map(chunk => chunk.text).join('\n')

    // 根据当前模式更新对应的编辑器
    if (editorMode === 'plaintext') {
      setPlainTextContent(mergedText)
    } else {
      // Word模式：使用SuperDoc的API设置内容
      const instance = editorRef.current?.getInstance()
      if (instance && typeof instance.setContent === 'function') {
        instance.setContent(mergedText)
      } else if (instance && typeof instance.setText === 'function') {
        instance.setText(mergedText)
      }
    }

    // 清除检测结果
    setDetectionResult(null)
    setSelectedChunk(null)
  }, [detectionResult, editorMode])

  const handleExport = async (format: 'docx' | 'txt' | 'markdown') => {
    const instance = editorRef.current?.getInstance()
    if (!instance) return

    if (format === 'docx') {
      await instance.export({ triggerDownload: true })
    } else {
      const html = instance.getHTML()
      const div = window.document.createElement('div')
      div.innerHTML = html
      let content = div.textContent || div.innerText || ''

      if (format === 'markdown') {
        content = content
          .replace(/<h1[^>]*>(.*?)<\/h1>/gi, '# $1\n\n')
          .replace(/<h2[^>]*>(.*?)<\/h2>/gi, '## $1\n\n')
          .replace(/<h3[^>]*>(.*?)<\/h3>/gi, '### $1\n\n')
          .replace(/<b[^>]*>(.*?)<\/b>/gi, '**$1**')
          .replace(/<strong[^>]*>(.*?)<\/strong>/gi, '**$1**')
          .replace(/<i[^>]*>(.*?)<\/i>/gi, '*$1*')
          .replace(/<em[^>]*>(.*?)<\/em>/gi, '*$1*')
          .replace(/<br\s*\/?>/gi, '\n')
          .replace(/<p[^>]*>(.*?)<\/p>/gi, '$1\n\n')
          .replace(/<[^>]+>/g, '')
      }

      const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
      const url = URL.createObjectURL(blob)
      const a = window.document.createElement('a')
      a.href = url
      a.download = `document.${format === 'markdown' ? 'md' : format}`
      a.click()
      URL.revokeObjectURL(url)
    }
  }

  const renderPreview = () => {
    if (!detectionResult || !detectionResult.chunks.length) {
      return (
        <div className="preview-document">
          <p className="empty-message">
            点击「开始检测」查看分析结果
          </p>
        </div>
      )
    }

    return (
      <div className="preview-document">
        {detectionResult.chunks.map((chunk, index) => (
          <TextChunk
            key={index}
            chunk={chunk}
            index={index}
            isJumping={jumpingChunk === index}
            isSelected={selectedChunk === index}
            isEditing={editingChunk === index}
            isDetecting={detectingChunk === index}
            onSelect={(idx) => {
              setSelectedChunk(idx)
              scrollToChunk(idx)
            }}
            onEdit={(idx) => setEditingChunk(idx)}
            onCancelEdit={() => setEditingChunk(null)}
            onReDetect={(idx, text) => handleChunkDetect(idx, text)}
            onJumpingChange={(idx) => setJumpingChunk(idx)}
          />
        ))}
      </div>
    )
  }

  return (
    <div className="app-container">
      {/* 左侧工具栏 */}
      <aside className="sidebar" aria-label="工具栏">
        <ToolbarButton
          icon={<IconFile />}
          label="打开文件"
          ariaLabel="打开文件"
          onClick={() => fileInputRef.current?.click()}
        />
        <input
          ref={fileInputRef}
          type="file"
          accept=".docx,.txt"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) setDocFile(file)
          }}
          aria-label="选择要检测的文件"
          hidden
        />

        <div className="sidebar-divider" />

        <ToolbarButton
          icon={<IconChart />}
          label="统计"
          ariaLabel="查看检测统计"
          active={!!detectionResult}
        />

        <div className="sidebar-divider" />

        <ToolbarButton
          icon={<IconDownload />}
          label="导出"
          ariaLabel="导出文档"
          showDropdown
          dropdownContent={
            <>
              <button className="dropdown-item" role="menuitem" onClick={() => handleExport('docx')}>
                Word文档
              </button>
              <button className="dropdown-item" role="menuitem" onClick={() => handleExport('txt')}>
                纯文本
              </button>
              <button className="dropdown-item" role="menuitem" onClick={() => handleExport('markdown')}>
                Markdown
              </button>
            </>
          }
        />

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
        <button
          className="theme-toggle"
          onClick={toggleTheme}
          aria-label={theme === 'light' ? "切换到深色模式" : "切换到浅色模式"}
        >
          {theme === 'light' ? <IconMoon /> : <IconSun />}
        </button>
      </aside>

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
        {/* 编辑面板 */}
        <div className="editor-panel" style={{ width: `${splitRatio * 100}%` }}>
          <div className="panel-header">
            <span className="panel-title">文档编辑</span>
            <div className="detection-bar">
              {/* 字数显示 */}
              <span className="word-count-display">
                {editorMode === 'plaintext' ? plainTextContent.length : editorPlainText.length} 字
              </span>

              {/* 分块大小选择器 */}
              <select
                value={chunkSize}
                onChange={(e) => setChunkSize(e.target.value)}
                className="model-select"
                aria-label="分块大小"
              >
                <option value="original">原文</option>
                <option value="200">200字/块</option>
                <option value="500">500字/块</option>
                <option value="1000">1000字/块</option>
              </select>

              {/* 检测按钮 */}
              <button className="btn btn-primary" onClick={handleDetect} disabled={isDetecting}>
                {isDetecting ? '检测中...' : (detectionResult ? '重新检测' : '开始检测')}
              </button>
            </div>
          </div>
          {error && (
            <div className="error-banner" role="alert">
              {error}
            </div>
          )}
          <div className="panel-content">
            {editorMode === 'plaintext' ? (
              <PlainTextEditor
                value={plainTextContent}
                onChange={setPlainTextContent}
              />
            ) : docFile ? (
              <>
                <SuperDocEditor
                  ref={editorRef}
                  document={docFile}
                  documentMode={mode}
                  role="editor"
                  user={{ name: 'User', email: 'user@example.com' }}
                  rulers
                  onReady={() => setIsReady(true)}
                  style={{ height: '100%' }}
                />
                {isDetecting && (
                  <div className="loading-overlay">
                    <div className="spinner" />
                  </div>
                )}
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
          {/* 左侧状态栏 - 项目信息 */}
          <div className="status-bar left-status-bar">
            <div className="status-item">
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="status-link"
                title="访问 GitHub 仓库"
              >
                <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
                  <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
                </svg>
                <span>AIGC Detector</span>
              </a>
            </div>
            <div className="status-item">
              <span className="status-version">v1.0.0</span>
            </div>
          </div>
        </div>

        {/* 可拖拽分隔条 */}
        <div
          className="panel-resizer"
          onMouseDown={(e) => {
            e.preventDefault()
            const startX = e.clientX
            const startRatio = splitRatio
            const containerWidth = e.currentTarget.parentElement?.offsetWidth || 1

            const onMouseMove = (e: MouseEvent) => {
              const delta = e.clientX - startX
              const newRatio = startRatio + delta / containerWidth
              setSplitRatio(Math.max(0.2, Math.min(0.8, newRatio)))
            }

            const onMouseUp = () => {
              document.removeEventListener('mousemove', onMouseMove)
              document.removeEventListener('mouseup', onMouseUp)
            }

            document.addEventListener('mousemove', onMouseMove, { passive: true })
            document.addEventListener('mouseup', onMouseUp)
          }}
        />

        {/* 预览面板 */}
        <div className="preview-panel" style={{ width: `${(1 - splitRatio) * 100}%` }}>
          <div className="panel-header">
            <span className="panel-title">检测结果</span>
            {detectionResult && (
              <>
                <div className="overall-probability" role="status" aria-live="polite">
                  <span>AI:</span>
                  <div className="probability-bar" role="progressbar" aria-valuenow={Math.round(detectionResult.overall_probability * 100)} aria-valuemin={0} aria-valuemax={100}>
                    <div
                      className="probability-fill"
                      style={{ width: `${detectionResult.overall_probability * 100}%` }}
                    />
                  </div>
                  <span>{(detectionResult.overall_probability * 100).toFixed(1)}%</span>
                </div>
                <button
                  className="btn btn-primary merge-btn"
                  onClick={handleMergeToOriginal}
                  title="将编辑后的文本合并并应用到原文"
                >
                  合并到原文
                </button>
              </>
            )}
          </div>
          <div className="preview-content">
            {renderPreview()}
          </div>
          <div className="status-bar right-status-bar">
            <div className="status-item">
              <span className={`status-dot ${isReady || editorMode === 'plaintext' ? 'ready' : 'loading'}`} />
              <span>{isReady || editorMode === 'plaintext' ? '就绪' : '加载中...'}</span>
            </div>
            {detectionResult && (
              <div className="status-item">
                <span>{detectionResult.text_length} 字</span>
                <span>·</span>
                <span>{detectionResult.chunks.length} 个段落</span>
              </div>
            )}
            {/* 系统状态信息 */}
            {systemInfo && (
              <>
                <div className="status-divider" />
                <div className="status-item" title={`CPU: ${systemInfo.cpu.count} 核心`}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="12" height="12">
                    <rect x="4" y="4" width="16" height="16" rx="2" />
                    <rect x="9" y="9" width="6" height="6" />
                    <line x1="9" y1="1" x2="9" y2="4" />
                    <line x1="15" y1="1" x2="15" y2="4" />
                    <line x1="9" y1="20" x2="9" y2="23" />
                    <line x1="15" y1="20" x2="15" y2="23" />
                    <line x1="20" y1="9" x2="23" y2="9" />
                    <line x1="20" y1="14" x2="23" y2="14" />
                    <line x1="1" y1="9" x2="4" y2="9" />
                    <line x1="1" y1="14" x2="4" y2="14" />
                  </svg>
                  <span>{systemInfo.cpu.percent.toFixed(0)}%</span>
                </div>
                <div className="status-item" title={`内存: ${systemInfo.memory.used_gb} / ${systemInfo.memory.total_gb} GB`}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="12" height="12">
                    <rect x="2" y="6" width="20" height="12" rx="2" />
                    <line x1="6" y1="10" x2="6" y2="14" />
                    <line x1="10" y1="10" x2="10" y2="14" />
                    <line x1="14" y1="10" x2="14" y2="14" />
                    <line x1="18" y1="10" x2="18" y2="14" />
                  </svg>
                  <span>{systemInfo.memory.percent.toFixed(0)}%</span>
                </div>
                {systemInfo.gpu && (
                  <div className="status-item gpu-status" title={`GPU: ${systemInfo.gpu.name}`}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="12" height="12">
                      <rect x="2" y="6" width="20" height="12" rx="2" />
                      <circle cx="8" cy="12" r="2" />
                      <circle cx="16" cy="12" r="2" />
                    </svg>
                    <span>{systemInfo.gpu.memory_allocated_gb.toFixed(1)}G</span>
                  </div>
                )}
                <div className="status-item" title={systemInfo.model.loaded ? '模型已加载' : '模型加载中...'}>
                  <span className={`status-dot ${systemInfo.model.loaded ? 'ready' : 'loading'}`} />
                  <span>{systemInfo.model.loaded ? '模型就绪' : '模型加载中'}</span>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
