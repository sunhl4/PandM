import { useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import MarkdownIt from 'markdown-it'
import { NODE_TYPE_CONFIG } from '../data/initialGraph.js'

const md = new MarkdownIt({ html: false, linkify: true, typographer: true })

const TYPE_LABELS = {
  root: '根节点',
  module: '模块',
  category: '分类',
  topic: '主题',
  leaf: '资料',
}

function TagBadge({ tag }) {
  return (
    <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium"
      style={{ background: 'rgba(0,212,255,0.1)', color: '#00d4ff', border: '1px solid rgba(0,212,255,0.25)' }}>
      {tag}
    </span>
  )
}

function ChildNodeItem({ node, onClick }) {
  const cfg = NODE_TYPE_CONFIG[node.type] || NODE_TYPE_CONFIG.leaf
  return (
    <button
      onClick={() => onClick(node)}
      className="w-full text-left flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all group"
      style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}
      onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(0,212,255,0.06)'; e.currentTarget.style.borderColor = 'rgba(0,212,255,0.2)' }}
      onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.03)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)' }}
    >
      <div className="flex-shrink-0 w-2.5 h-2.5 rounded-full"
        style={{ background: cfg.color, boxShadow: `0 0 6px ${cfg.color}` }} />
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium truncate" style={{ color: '#e2e8f0' }}>{node.label}</div>
        {node.description && (
          <div className="text-xs truncate mt-0.5" style={{ color: '#64748b' }}>{node.description}</div>
        )}
      </div>
      <svg className="w-3.5 h-3.5 flex-shrink-0 opacity-40 group-hover:opacity-80 transition-opacity"
        fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
      </svg>
    </button>
  )
}

export default function NodePanel({ node, children, ancestors, onClose, onSelectNode, onAddNode, onFocusChange, onDeleteNode }) {
  const panelRef = useRef(null)
  if (!node) return null

  const cfg = NODE_TYPE_CONFIG[node.type] || NODE_TYPE_CONFIG.leaf
  const htmlContent = node.content ? md.render(node.content) : ''

  return (
    <AnimatePresence>
      <motion.div
        key={node.id}
        ref={panelRef}
        initial={{ x: 380, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: 380, opacity: 0 }}
        transition={{ type: 'spring', stiffness: 320, damping: 32 }}
        className="absolute right-0 top-0 h-full w-96 flex flex-col overflow-hidden z-20"
        style={{
          background: 'rgba(7,7,15,0.92)',
          backdropFilter: 'blur(20px)',
          borderLeft: `1px solid rgba(${hexToRgb(cfg.color)},0.25)`,
          boxShadow: `-8px 0 40px rgba(0,0,0,0.5)`,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex-shrink-0 px-5 pt-5 pb-4"
          style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 rounded-full flex-shrink-0"
                style={{ background: cfg.color, boxShadow: `0 0 10px ${cfg.color}` }} />
              <div>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-xs font-medium px-2 py-0.5 rounded"
                    style={{ background: `rgba(${hexToRgb(cfg.color)},0.15)`, color: cfg.color }}>
                    {TYPE_LABELS[node.type] || node.type}
                  </span>
                </div>
                <h2 className="text-lg font-semibold mt-1.5 leading-tight" style={{ color: '#f1f5f9' }}>
                  {node.label}
                </h2>
                {node.description && (
                  <p className="text-xs mt-1 leading-relaxed" style={{ color: '#64748b' }}>
                    {node.description}
                  </p>
                )}
              </div>
            </div>
            <button
              onClick={onClose}
              className="flex-shrink-0 p-1.5 rounded-lg transition-all hover:bg-white/10"
              style={{ color: '#64748b' }}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Tags */}
          {node.tags?.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-3">
              {node.tags.map((tag) => <TagBadge key={tag} tag={tag} />)}
            </div>
          )}

          {/* Focus button */}
          {node.id !== 'root' && (
            <button
              onClick={() => onFocusChange(node.id)}
              className="mt-3 w-full py-1.5 rounded-lg text-xs font-medium transition-all flex items-center justify-center gap-2"
              style={{ background: `rgba(${hexToRgb(cfg.color)},0.12)`, color: cfg.color, border: `1px solid rgba(${hexToRgb(cfg.color)},0.25)` }}
              onMouseEnter={(e) => { e.currentTarget.style.background = `rgba(${hexToRgb(cfg.color)},0.22)` }}
              onMouseLeave={(e) => { e.currentTarget.style.background = `rgba(${hexToRgb(cfg.color)},0.12)` }}
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
              聚焦此节点子图
            </button>
          )}
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">
          {/* Markdown content */}
          {htmlContent && (
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider mb-2.5"
                style={{ color: '#475569' }}>内容</div>
              <div
                className="markdown-body text-sm"
                dangerouslySetInnerHTML={{ __html: htmlContent }}
              />
            </div>
          )}

          {/* Links */}
          {node.links?.length > 0 && (
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider mb-2.5"
                style={{ color: '#475569' }}>相关文件</div>
              <div className="space-y-1.5">
                {node.links.map((link, i) => (
                  <a
                    key={i}
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-xs py-1.5 px-2 rounded transition-all"
                    style={{ color: '#00d4ff' }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(0,212,255,0.08)' }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
                  >
                    <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                    {link.label}
                  </a>
                ))}
              </div>
            </div>
          )}

          {/* Children */}
          {children?.length > 0 && (
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider mb-2.5"
                style={{ color: '#475569' }}>
                子节点 ({children.length})
              </div>
              <div className="space-y-1.5">
                {children.map((child) => (
                  <ChildNodeItem key={child.id} node={child} onClick={onSelectNode} />
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer: Add buttons + Delete */}
        <div className="flex-shrink-0 px-5 py-4 space-y-2"
          style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => onAddNode(node.id, 'category')}
              className="py-2 rounded-lg text-xs font-medium transition-all"
              style={{ background: 'rgba(139,92,246,0.12)', color: '#8b5cf6', border: '1px solid rgba(139,92,246,0.25)' }}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(139,92,246,0.22)' }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(139,92,246,0.12)' }}
            >
              + 新建分支
            </button>
            <button
              onClick={() => onAddNode(node.id, 'leaf')}
              className="py-2 rounded-lg text-xs font-medium transition-all"
              style={{ background: 'rgba(16,185,129,0.12)', color: '#10b981', border: '1px solid rgba(16,185,129,0.25)' }}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(16,185,129,0.22)' }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(16,185,129,0.12)' }}
            >
              + 添加资料
            </button>
          </div>
          {node.id !== 'root' && node.id !== 'work-plan' && node.id !== 'learning' && (
            <button
              onClick={() => { onDeleteNode(node.id); onClose() }}
              className="w-full py-1.5 rounded-lg text-xs transition-all"
              style={{ color: '#ef4444', background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.15)' }}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(239,68,68,0.14)' }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(239,68,68,0.06)' }}
            >
              删除节点
            </button>
          )}
        </div>
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
