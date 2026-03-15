import { useState, useRef, useCallback } from 'react'
import { SuperDocEditor } from '@superdoc-dev/react'
import type { SuperDocRef } from '@superdoc-dev/react'
import '@superdoc-dev/react/style.css'

// 类型定义
interface ChunkDetection {
  text: string
  probability: number
  index: number
  text_length: number
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

// 主应用组件
function App() {
  // 状态
  const [docFile, setDocFile] = useState<File | null>(null)
  const [mode] = useState<'editing' | 'viewing' | 'suggesting'>('editing')
  const [isReady, setIsReady] = useState(false)
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null)
  const [isDetecting, setIsDetecting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [chunkSize, setChunkSize] = useState<string>('original')
  const [selectedChunk, setSelectedChunk] = useState<number | null>(null)
  const [editorPlainText, setEditorPlainText] = useState<string>('')

  const editorRef = useRef<SuperDocRef>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // 跳转到指定块并在编辑器中高亮显示
  const scrollToChunk = useCallback((index: number) => {
    const instance = editorRef.current?.getInstance()
    if (!instance || !detectionResult?.chunks[index]) return

    const chunk = detectionResult.chunks[index]
    const chunkText = chunk.text

    try {
      // 使用搜索功能查找文本并高亮
      // @ts-ignore - SuperDoc可能有search方法
      const searchResult = instance.search?.(chunkText, { highlight: true })

      if (searchResult && searchResult.length > 0) {
        // 跳转到第一个匹配结果
        // @ts-ignore
        instance.goToSearchResult?.(searchResult[0])
        // 聚焦编辑器
        // @ts-ignore
        instance.focus?.()
      }
    } catch (e) {
      console.error('跳转失败:', e)
    }
  }, [detectionResult])

  // 获取编辑器文本并检测
  const handleDetect = useCallback(async () => {
    const instance = editorRef.current?.getInstance()
    if (!instance) return

    setIsDetecting(true)
    setError(null)
    try {
      // 获取纯文本
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

      // 保存原始文本用于后续跳转
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

  // 导出功能
  const handleExport = async (format: 'docx' | 'txt' | 'markdown') => {
    const instance = editorRef.current?.getInstance()
    if (!instance) return

    if (format === 'docx') {
      await instance.export({ triggerDownload: true })
    } else {
      // 获取HTML并转换
      const html = instance.getHTML()
      const div = window.document.createElement('div')
      div.innerHTML = html
      let content = div.textContent || div.innerText || ''

      if (format === 'markdown') {
        // 简单HTML到Markdown转换
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

      // 下载文件
      const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
      const url = URL.createObjectURL(blob)
      const a = window.document.createElement('a')
      a.href = url
      a.download = `document.${format === 'markdown' ? 'md' : format}`
      a.click()
      URL.revokeObjectURL(url)
    }
  }

  // 渲染预览 - 带AI概率显示和编辑功能
  const renderPreview = () => {
    if (!detectionResult || !detectionResult.chunks.length) {
      return (
        <div className="preview-document">
          <p style={{ color: '#999', textAlign: 'center', marginTop: '100px' }}>
            点击"开始检测"查看预览效果
          </p>
        </div>
      )
    }

    return (
      <div className="preview-document" style={{ padding: '20px' }}>
        {detectionResult.chunks.map((chunk, index) => (
          <div
            key={index}
            className={`text-chunk ${selectedChunk === index ? 'selected' : ''}`}
            style={{
              backgroundColor:
                chunk.probability > 0.7 ? 'rgba(244, 67, 54, 0.15)' :
                chunk.probability > 0.4 ? 'rgba(255, 152, 0, 0.15)' :
                chunk.probability > 0.2 ? 'rgba(255, 193, 7, 0.15)' :
                'rgba(76, 175, 80, 0.15)',
              borderLeft: `4px solid ${
                chunk.probability > 0.7 ? '#f44336' :
                chunk.probability > 0.4 ? '#ff9800' :
                chunk.probability > 0.2 ? '#ffc107' :
                '#4caf50'
              }`,
              padding: '12px 16px',
              margin: '8px 0',
              borderRadius: '6px',
              cursor: 'pointer',
              display: 'block',
              width: '100%',
              boxShadow: selectedChunk === index ? '0 0 0 2px var(--theme-accent)' : 'none',
            }}
            onClick={() => {
              setSelectedChunk(index)
              scrollToChunk(index)
            }}
          >
            {/* AI概率徽章 - 放在顶部 */}
            <div style={{ marginBottom: '8px' }}>
              <span
                style={{
                  display: 'inline-block',
                  fontSize: '12px',
                  padding: '3px 10px',
                  borderRadius: '12px',
                  backgroundColor:
                    chunk.probability > 0.7 ? '#f44336' :
                    chunk.probability > 0.4 ? '#ff9800' :
                    chunk.probability > 0.2 ? '#ffc107' :
                    '#4caf50',
                  color: '#fff',
                  fontWeight: 600,
                }}
              >
                块{index + 1}: {Math.round(chunk.probability * 100)}% AI
              </span>
              <span style={{ marginLeft: '10px', fontSize: '12px', color: '#666' }}>
                {chunk.text_length}字
              </span>
            </div>
            {/* 文本内容 */}
            <div style={{
              fontSize: '15px',
              lineHeight: '1.8',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              fontFamily: 'var(--font-serif)'
            }}>
              {chunk.text}
            </div>
          </div>
        ))}
      </div>
    )
  }

  // 处理块编辑 - 重新检测单个块
  const handleChunkEdit = useCallback(async (index: number, newText: string) => {
    if (!detectionResult) return

    const newChunks = [...detectionResult.chunks]
    newChunks[index] = { ...newChunks[index], text: newText }
    setDetectionResult({ ...detectionResult, chunks: newChunks })
  }, [detectionResult])

  return (
    <div className="app-container">
      {/* 左侧工具栏 */}
      <aside className="sidebar" aria-label="工具栏">
        <ToolbarButton
          icon={<span aria-hidden="true">📄</span>}
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
          icon={<span aria-hidden="true">🔍</span>}
          label="检测"
          ariaLabel={isDetecting ? "检测中" : "开始AI检测"}
          onClick={handleDetect}
          disabled={isDetecting}
        />

        <ToolbarButton
          icon={<span aria-hidden="true">📊</span>}
          label="统计"
          ariaLabel="查看检测统计"
          active={!!detectionResult}
        />

        <div className="sidebar-divider" aria-hidden="true" />

        <ToolbarButton
          icon={<span aria-hidden="true">💾</span>}
          label="导出"
          ariaLabel="导出文档"
          showDropdown
          dropdownContent={
            <>
              <button className="dropdown-item" role="menuitem" onClick={() => handleExport('docx')}>
                <span aria-hidden="true">📝</span> Word文档 (.docx)
              </button>
              <button className="dropdown-item" role="menuitem" onClick={() => handleExport('txt')}>
                <span aria-hidden="true">📄</span> 纯文本 (.txt)
              </button>
              <button className="dropdown-item" role="menuitem" onClick={() => handleExport('markdown')}>
                <span aria-hidden="true">📋</span> Markdown (.md)
              </button>
            </>
          }
        />
      </aside>

      {/* 主区域 */}
      <div className="main-area">
        {/* 编辑面板 */}
        <div className="editor-panel">
          <div className="panel-header">
            <span className="panel-title">编辑区</span>
            <div className="detection-bar">
              {detectionResult && (
                <>
                  <div className="overall-probability" role="status" aria-live="polite">
                    <span>AI概率:</span>
                    <div className="probability-bar" role="progressbar" aria-valuenow={Math.round(detectionResult.overall_probability * 100)} aria-valuemin={0} aria-valuemax={100} aria-label="AI生成概率">
                      <div
                        className="probability-fill"
                        style={{ width: `${detectionResult.overall_probability * 100}%` }}
                        aria-hidden="true"
                      />
                    </div>
                    <span>{(detectionResult.overall_probability * 100).toFixed(1)}%</span>
                  </div>
                  <span className="sr-only">
                    检测完成，AI生成概率为{(detectionResult.overall_probability * 100).toFixed(1)}%，
                    共{detectionResult.chunks.length}个段落
                  </span>
                </>
              )}
              <button className="btn btn-primary" onClick={handleDetect} disabled={isDetecting} aria-busy={isDetecting}>
                {isDetecting ? '检测中...' : '开始检测'}
              </button>
            </div>
          </div>
          {error && (
            <div role="alert" style={{ padding: '8px 16px', background: '#fee2e2', color: '#dc2626', borderBottom: '1px solid #fecaca' }}>
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
            <span className="panel-title">检测结果预览</span>
            {/* 分块大小选择器 */}
            <div className="detection-bar">
              <select
                value={chunkSize}
                onChange={(e) => setChunkSize(e.target.value)}
                className="model-select"
                aria-label="分块大小"
                style={{ marginRight: '8px' }}
              >
                <option value="original">原文</option>
                <option value="200">200字/块</option>
                <option value="500">500字/块</option>
                <option value="1000">1000字/块</option>
              </select>
              <button
                className="btn btn-primary"
                onClick={handleDetect}
                disabled={isDetecting}
                style={{ padding: '6px 12px', fontSize: '12px' }}
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
                <span>字数: {detectionResult.text_length}</span>
                <span>•</span>
                <span>段落: {detectionResult.chunks.length}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
