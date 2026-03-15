import { useState, useRef, useCallback, useEffect } from 'react'
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

// 类型定义
interface ChunkDetection {
  text: string
  probability: number
  index: number
  text_length: number
  start_pos: number
}

interface DetectionResult {
  chunks: ChunkDetection[]
  overall_probability: number
  text_length: number
  mode: string
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
    <div className="dropdown" style={{ position: 'relative' }}>
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
  const [theme, setTheme] = useState<'light' | 'dark'>('light')

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

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    setTheme(newTheme)
    localStorage.setItem('theme', newTheme)
    document.documentElement.setAttribute('data-theme', newTheme)
  }

  // 文本规范化
  const normalizeText = (text: string): string => {
    return text.replace(/[\s\n\r]+/g, ' ').trim()
  }

  const fuzzyMatch = (source: string, target: string): number => {
    const s = normalizeText(source)
    const t = normalizeText(target)
    if (!s || !t) return 0
    const minLen = Math.min(s.length, t.length)
    const maxLen = Math.max(s.length, t.length)
    if (maxLen === 0) return 1
    let matches = 0
    for (let i = 0; i < minLen; i++) {
      if (s[i] === t[i]) matches++
    }
    return matches / maxLen
  }

  const scrollToChunk = useCallback((index: number) => {
    const instance = editorRef.current?.getInstance()
    if (!instance || !detectionResult) return

    const chunk = detectionResult.chunks[index]
    if (!chunk) return

    const chunkStartPos = chunk.start_pos || 0
    const chunkText = chunk.text.trim()

    let editorText = ''
    if (typeof instance.getText === 'function') {
      editorText = instance.getText()
    }
    if (!editorText && editorPlainText) {
      editorText = editorPlainText
    }

    if (!editorText) return

    const searchRadius = 50
    let foundPos = -1
    const searchText = chunkText.substring(0, 50)

    for (let offset = 0; offset <= searchRadius; offset++) {
      const pos1 = chunkStartPos + offset
      if (pos1 + searchText.length <= editorText.length) {
        const candidate = editorText.substring(pos1, pos1 + searchText.length)
        if (fuzzyMatch(candidate, searchText) > 0.85) {
          foundPos = pos1
          break
        }
      }
      const pos2 = chunkStartPos - offset
      if (pos2 >= 0 && pos2 + searchText.length <= editorText.length) {
        const candidate = editorText.substring(pos2, pos2 + searchText.length)
        if (fuzzyMatch(candidate, searchText) > 0.85) {
          foundPos = pos2
          break
        }
      }
    }

    if (foundPos === -1) {
      const normalizedEditor = normalizeText(editorText)
      const normalizedChunk = normalizeText(searchText)
      foundPos = normalizedEditor.indexOf(normalizedChunk)
    }

    if (foundPos === -1) {
      // @ts-ignore
      if (instance.search && chunk.text) {
        const searchText2 = chunk.text.substring(0, 30).trim()
        if (searchText2.length >= 5) {
          try {
            // @ts-ignore
            const results = instance.search(searchText2, { highlight: true })
            if (results && results.length > 0) {
              // @ts-ignore
              instance.goToSearchResult?.(results[0])
              return
            }
          } catch (e) {}
        }
      }
    }

  }, [detectionResult, editorPlainText])

  const handleDetect = useCallback(async () => {
    const instance = editorRef.current?.getInstance()
    if (!instance) return

    setIsDetecting(true)
    setError(null)
    try {
      let plainText = ''
      if (typeof instance.getText === 'function') {
        plainText = instance.getText()
      }
      if (!plainText || plainText.length < 10) {
        const editorText = instance.getHTML()
        const div = window.document.createElement('div')
        div.innerHTML = editorText
        plainText = div.textContent || div.innerText || ''
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
  }, [chunkSize])

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
          <p style={{ color: 'var(--theme-text-muted)', textAlign: 'center', marginTop: '80px', fontSize: '14px' }}>
            点击「开始检测」查看分析结果
          </p>
        </div>
      )
    }

    return (
      <div className="preview-document">
        {detectionResult.chunks.map((chunk, index) => {
          const aiLevel = getAILevel(chunk.probability)
          const isJumping = jumpingChunk === index
          const isSelected = selectedChunk === index

          return (
            <div
              key={index}
              ref={(el) => {
                // 跳转完成后自动滚动到视图
                if (isJumping && el) {
                  setTimeout(() => {
                    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
                    setJumpingChunk(null)
                  }, 50)
                }
              }}
              className={`text-chunk ${aiLevel} ${isSelected ? 'selected' : ''} ${isJumping ? 'jump-to' : ''}`}
              onClick={() => {
                setSelectedChunk(index)
                setJumpingChunk(index)
                scrollToChunk(index)
              }}
            >
              <div>
                <span className={`ai-badge ${aiLevel}`}>
                  {Math.round(chunk.probability * 100)}% AI
                </span>
                <span className="chunk-meta">
                  {chunk.text_length} 字
                </span>
              </div>
              <div className="chunk-text">
                {chunk.text}
              </div>
            </div>
          )
        })}
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
          icon={<IconSearch />}
          label="检测"
          ariaLabel={isDetecting ? "检测中" : "开始AI检测"}
          onClick={handleDetect}
          disabled={isDetecting}
        />

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

        {/* 主题切换 */}
        <button
          className="theme-toggle"
          onClick={toggleTheme}
          aria-label={theme === 'light' ? "切换到深色模式" : "切换到浅色模式"}
        >
          {theme === 'light' ? <IconMoon /> : <IconSun />}
        </button>
      </aside>

      {/* 主区域 */}
      <div className="main-area">
        {/* 编辑面板 */}
        <div className="editor-panel">
          <div className="panel-header">
            <span className="panel-title">文档编辑</span>
            <div className="detection-bar">
              {detectionResult && (
                <>
                  <div className="overall-probability" role="status" aria-live="polite">
                    <span>AI概率:</span>
                    <div className="probability-bar" role="progressbar" aria-valuenow={Math.round(detectionResult.overall_probability * 100)} aria-valuemin={0} aria-valuemax={100}>
                      <div
                        className="probability-fill"
                        style={{ width: `${detectionResult.overall_probability * 100}%` }}
                      />
                    </div>
                    <span>{(detectionResult.overall_probability * 100).toFixed(1)}%</span>
                  </div>
                </>
              )}
              <button className="btn btn-primary" onClick={handleDetect} disabled={isDetecting}>
                {isDetecting ? '检测中...' : '开始检测'}
              </button>
            </div>
          </div>
          {error && (
            <div className="error-banner" role="alert">
              {error}
            </div>
          )}
          <div className="panel-content">
            {docFile ? (
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
        </div>

        {/* 预览面板 */}
        <div className="preview-panel">
          <div className="panel-header">
            <span className="panel-title">检测结果</span>
            <div className="detection-bar">
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
              <button
                className="btn btn-secondary"
                onClick={handleDetect}
                disabled={isDetecting}
              >
                {isDetecting ? '检测中...' : '重新检测'}
              </button>
            </div>
          </div>
          <div className="preview-content">
            {renderPreview()}
          </div>
          <div className="status-bar">
            <div className="status-item">
              <span className={`status-dot ${isReady ? 'ready' : 'loading'}`} />
              <span>{isReady ? '就绪' : '加载中...'}</span>
            </div>
            {detectionResult && (
              <div className="status-item">
                <span>{detectionResult.text_length} 字</span>
                <span>·</span>
                <span>{detectionResult.chunks.length} 个段落</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
