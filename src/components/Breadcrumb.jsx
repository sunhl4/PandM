import { NODE_TYPE_CONFIG } from '../data/initialGraph.js'

export default function Breadcrumb({ ancestors, currentNode, onNavigate }) {
  if (!currentNode) return null
  const allNodes = [...(ancestors || []), currentNode].filter(Boolean)
  if (allNodes.length <= 1 && currentNode?.id === 'root') return null

  return (
    <nav className="flex items-center gap-1 text-xs overflow-x-auto"
      style={{ maxWidth: '40vw', scrollbarWidth: 'none' }}>
      {allNodes.map((node, i) => {
        const cfg = NODE_TYPE_CONFIG[node.type] || NODE_TYPE_CONFIG.leaf
        const isLast = i === allNodes.length - 1
        return (
          <span key={node.id} className="flex items-center gap-1 flex-shrink-0">
            {i > 0 && (
              <svg className="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"
                style={{ color: '#334155' }}>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            )}
            <button
              onClick={() => !isLast && onNavigate(node)}
              disabled={isLast}
              className="flex items-center gap-1.5 px-2 py-1 rounded-lg transition-all"
              style={{
                color: isLast ? cfg.color : '#94a3b8',
                background: isLast ? `rgba(${hexToRgb(cfg.color)},0.1)` : 'transparent',
                border: isLast ? `1px solid rgba(${hexToRgb(cfg.color)},0.25)` : '1px solid transparent',
                cursor: isLast ? 'default' : 'pointer',
              }}
              onMouseEnter={(e) => { if (!isLast) e.currentTarget.style.color = '#e2e8f0' }}
              onMouseLeave={(e) => { if (!isLast) e.currentTarget.style.color = '#94a3b8' }}
            >
              <div className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                style={{ background: cfg.color }} />
              {node.label}
            </button>
          </span>
        )
      })}
    </nav>
  )
}

function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
  return result
    ? `${parseInt(result[1], 16)},${parseInt(result[2], 16)},${parseInt(result[3], 16)}`
    : '0,212,255'
}
