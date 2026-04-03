import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { NODE_TYPE_CONFIG } from '../data/initialGraph.js'

const TYPE_OPTIONS = [
  { value: 'module', label: '模块', desc: '大类模块' },
  { value: 'category', label: '分类', desc: '知识分类' },
  { value: 'topic', label: '主题', desc: '具体主题' },
  { value: 'leaf', label: '资料', desc: '具体资料/笔记' },
]

export default function AddNodeModal({ parentNode, defaultType, onConfirm, onCancel }) {
  const [label, setLabel] = useState('')
  const [type, setType] = useState(defaultType || 'topic')
  const [description, setDescription] = useState('')
  const [tags, setTags] = useState('')
  const [content, setContent] = useState('')
  const [step, setStep] = useState(1) // 1: basic info, 2: content
  const labelRef = useRef(null)

  useEffect(() => {
    labelRef.current?.focus()
  }, [])

  useEffect(() => {
    const handler = (e) => e.key === 'Escape' && onCancel()
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onCancel])

  const handleSubmit = () => {
    if (!label.trim()) return
    onConfirm({
      label: label.trim(),
      type,
      description: description.trim(),
      tags: tags.split(/[,，\s]+/).filter(Boolean),
      content: content.trim() || `# ${label.trim()}\n\n${description.trim()}`,
      links: [],
    })
  }

  const cfg = NODE_TYPE_CONFIG[type] || NODE_TYPE_CONFIG.leaf
  const parentCfg = parentNode ? (NODE_TYPE_CONFIG[parentNode.type] || NODE_TYPE_CONFIG.leaf) : null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        style={{ background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)' }}
        onClick={onCancel}
      >
        <motion.div
          initial={{ scale: 0.92, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.92, opacity: 0, y: 20 }}
          transition={{ type: 'spring', stiffness: 380, damping: 30 }}
          className="w-full max-w-lg rounded-2xl overflow-hidden"
          style={{
            background: 'rgba(10,10,20,0.97)',
            border: '1px solid rgba(0,212,255,0.2)',
            boxShadow: '0 0 60px rgba(0,212,255,0.1), 0 20px 60px rgba(0,0,0,0.6)',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Modal header */}
          <div className="px-6 pt-6 pb-4" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold gradient-text">新建节点</h3>
                {parentNode && (
                  <div className="flex items-center gap-1.5 mt-1 text-xs" style={{ color: '#64748b' }}>
                    <span>添加到</span>
                    <div className="w-1.5 h-1.5 rounded-full" style={{ background: parentCfg?.color }} />
                    <span style={{ color: parentCfg?.color }}>{parentNode.label}</span>
                  </div>
                )}
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs px-2 py-1 rounded" style={{ background: 'rgba(255,255,255,0.05)', color: '#475569' }}>
                  {step}/2
                </span>
                <button onClick={onCancel} className="p-1.5 rounded-lg hover:bg-white/10 transition-all"
                  style={{ color: '#64748b' }}>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
          </div>

          <div className="px-6 py-5 space-y-4">
            {step === 1 && (
              <>
                {/* Node type selector */}
                <div>
                  <label className="block text-xs font-medium mb-2" style={{ color: '#94a3b8' }}>节点类型</label>
                  <div className="grid grid-cols-4 gap-2">
                    {TYPE_OPTIONS.map((opt) => {
                      const optCfg = NODE_TYPE_CONFIG[opt.value]
                      const isSelected = type === opt.value
                      return (
                        <button
                          key={opt.value}
                          onClick={() => setType(opt.value)}
                          className="p-2.5 rounded-xl text-center transition-all"
                          style={{
                            background: isSelected ? `rgba(${hexToRgb(optCfg.color)},0.15)` : 'rgba(255,255,255,0.03)',
                            border: `1px solid ${isSelected ? `rgba(${hexToRgb(optCfg.color)},0.4)` : 'rgba(255,255,255,0.07)'}`,
                            boxShadow: isSelected ? `0 0 12px rgba(${hexToRgb(optCfg.color)},0.2)` : 'none',
                          }}
                        >
                          <div className="w-2 h-2 rounded-full mx-auto mb-1.5"
                            style={{ background: optCfg.color, boxShadow: isSelected ? `0 0 6px ${optCfg.color}` : 'none' }} />
                          <div className="text-xs font-medium" style={{ color: isSelected ? optCfg.color : '#94a3b8' }}>
                            {opt.label}
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* Label */}
                <div>
                  <label className="block text-xs font-medium mb-1.5" style={{ color: '#94a3b8' }}>节点名称 *</label>
                  <input
                    ref={labelRef}
                    type="text"
                    value={label}
                    onChange={(e) => setLabel(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && label.trim() && setStep(2)}
                    placeholder="输入节点名称..."
                    className="w-full px-3 py-2.5 rounded-xl text-sm outline-none transition-all"
                    style={{
                      background: 'rgba(255,255,255,0.04)',
                      border: `1px solid ${label ? `rgba(${hexToRgb(cfg.color)},0.4)` : 'rgba(255,255,255,0.1)'}`,
                      color: '#e2e8f0',
                      caretColor: cfg.color,
                    }}
                  />
                </div>

                {/* Description */}
                <div>
                  <label className="block text-xs font-medium mb-1.5" style={{ color: '#94a3b8' }}>简短描述</label>
                  <input
                    type="text"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="一句话描述..."
                    className="w-full px-3 py-2.5 rounded-xl text-sm outline-none transition-all"
                    style={{
                      background: 'rgba(255,255,255,0.04)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      color: '#e2e8f0',
                    }}
                  />
                </div>

                {/* Tags */}
                <div>
                  <label className="block text-xs font-medium mb-1.5" style={{ color: '#94a3b8' }}>
                    标签 <span style={{ color: '#475569' }}>(逗号分隔)</span>
                  </label>
                  <input
                    type="text"
                    value={tags}
                    onChange={(e) => setTags(e.target.value)}
                    placeholder="例: VQE, 量子化学, NISQ"
                    className="w-full px-3 py-2.5 rounded-xl text-sm outline-none transition-all"
                    style={{
                      background: 'rgba(255,255,255,0.04)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      color: '#e2e8f0',
                    }}
                  />
                </div>
              </>
            )}

            {step === 2 && (
              <div>
                <label className="block text-xs font-medium mb-1.5" style={{ color: '#94a3b8' }}>
                  内容 <span style={{ color: '#475569' }}>(Markdown)</span>
                </label>
                <textarea
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  placeholder={`# ${label}\n\n在这里写入详细内容...`}
                  rows={10}
                  className="w-full px-3 py-2.5 rounded-xl text-sm outline-none transition-all resize-none font-mono"
                  style={{
                    background: 'rgba(255,255,255,0.04)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    color: '#e2e8f0',
                    fontSize: '12px',
                    lineHeight: '1.6',
                  }}
                />
                <p className="text-xs mt-1.5" style={{ color: '#475569' }}>支持 Markdown 格式，可留空（系统会自动生成默认内容）</p>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="px-6 pb-6 flex items-center justify-between gap-3">
            <button
              onClick={step === 1 ? onCancel : () => setStep(1)}
              className="px-4 py-2 rounded-xl text-sm transition-all"
              style={{ color: '#64748b', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.08)' }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.04)' }}
            >
              {step === 1 ? '取消' : '← 返回'}
            </button>
            <button
              onClick={step === 1 ? () => label.trim() && setStep(2) : handleSubmit}
              disabled={!label.trim()}
              className="px-5 py-2 rounded-xl text-sm font-medium transition-all disabled:opacity-40"
              style={{
                background: `linear-gradient(135deg, ${cfg.color}33, ${cfg.color}22)`,
                color: cfg.color,
                border: `1px solid ${cfg.color}66`,
                boxShadow: label.trim() ? `0 0 16px ${cfg.color}22` : 'none',
              }}
              onMouseEnter={(e) => { if (label.trim()) e.currentTarget.style.background = `linear-gradient(135deg, ${cfg.color}55, ${cfg.color}33)` }}
              onMouseLeave={(e) => { e.currentTarget.style.background = `linear-gradient(135deg, ${cfg.color}33, ${cfg.color}22)` }}
            >
              {step === 1 ? '下一步 →' : '创建节点 ✦'}
            </button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}

function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
  return result
    ? `${parseInt(result[1], 16)},${parseInt(result[2], 16)},${parseInt(result[3], 16)}`
    : '0,212,255'
}
