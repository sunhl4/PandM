import { useState, useMemo, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { INITIAL_GRAPH } from './data/initialGraph.js'
import { useLocalStorage } from './hooks/useLocalStorage.js'
import { useGraph } from './hooks/useGraph.js'
import ParticleField from './components/ParticleField.jsx'
import QuantumGraph from './components/QuantumGraph.jsx'
import NodePanel from './components/NodePanel.jsx'
import SearchBar from './components/SearchBar.jsx'
import Breadcrumb from './components/Breadcrumb.jsx'
import AddNodeModal from './components/AddNodeModal.jsx'

export default function App() {
  const [graph, setGraph] = useLocalStorage(INITIAL_GRAPH)
  const [selectedNode, setSelectedNode] = useState(null)
  const [focusNodeId, setFocusNodeId] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [addModal, setAddModal] = useState(null) // { parentId, defaultType }
  const [showLegend, setShowLegend] = useState(false)

  const { getNode, getChildren, getAncestors, addNode, updateNode, deleteNode } = useGraph(graph, setGraph)

  const focusNode = focusNodeId ? getNode(focusNodeId) : getNode('root')

  // Compute search matches
  const searchResultCount = useMemo(() => {
    if (!searchQuery.trim()) return graph.nodes.length
    const q = searchQuery.toLowerCase()
    return graph.nodes.filter((n) =>
      n.label.toLowerCase().includes(q) ||
      (n.tags || []).some((t) => t.toLowerCase().includes(q)) ||
      (n.description || '').toLowerCase().includes(q)
    ).length
  }, [searchQuery, graph.nodes])

  const handleSelectNode = useCallback((node) => {
    setSelectedNode(node)
  }, [])

  const handleFocusChange = useCallback((nodeId) => {
    setFocusNodeId(nodeId === 'root' ? null : nodeId)
    setSelectedNode(getNode(nodeId))
  }, [getNode])

  const handleBreadcrumbNavigate = useCallback((node) => {
    if (node.id === 'root') {
      setFocusNodeId(null)
    } else {
      setFocusNodeId(node.id)
    }
    setSelectedNode(node)
  }, [])

  const handleAddNode = useCallback((parentId, defaultType) => {
    setAddModal({ parentId, defaultType })
  }, [])

  const handleConfirmAdd = useCallback((nodeData) => {
    if (!addModal) return
    const newId = addNode(addModal.parentId, nodeData)
    setAddModal(null)
    setTimeout(() => setSelectedNode(getNode(newId)), 100)
  }, [addModal, addNode, getNode])

  const handleDeleteNode = useCallback((id) => {
    if (selectedNode?.id === id) setSelectedNode(null)
    if (focusNodeId === id) setFocusNodeId(null)
    deleteNode(id)
  }, [selectedNode, focusNodeId, deleteNode])

  const handleResetGraph = useCallback(() => {
    if (window.confirm('确定要重置知识图谱吗？所有自定义内容将被清除。')) {
      localStorage.removeItem('pandm-graph-v1')
      window.location.reload()
    }
  }, [])

  const selectedChildren = selectedNode ? getChildren(selectedNode.id) : []
  const selectedAncestors = selectedNode ? getAncestors(selectedNode.id).map(getNode).filter(Boolean) : []
  const focusAncestors = focusNode ? getAncestors(focusNode.id).map(getNode).filter(Boolean) : []

  return (
    <div className="relative w-full h-full overflow-hidden" style={{ background: '#07070f' }}>
      {/* Animated background */}
      <ParticleField />

      {/* Subtle grid overlay */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px',
        }}
      />

      {/* Top header bar */}
      <header
        className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-5 py-3"
        style={{
          background: 'rgba(7,7,15,0.85)',
          backdropFilter: 'blur(16px)',
          borderBottom: '1px solid rgba(0,212,255,0.1)',
        }}
      >
        {/* Left: Logo + breadcrumb */}
        <div className="flex items-center gap-4 min-w-0">
          {/* Logo */}
          <div className="flex items-center gap-2.5 flex-shrink-0">
            <div className="relative w-7 h-7">
              <svg viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
                <circle cx="14" cy="14" r="5" fill="#00d4ff" fillOpacity="0.9" />
                <ellipse cx="14" cy="14" rx="13" ry="5" stroke="#00d4ff" strokeWidth="1.2" strokeOpacity="0.6" />
                <ellipse cx="14" cy="14" rx="13" ry="5" stroke="#8b5cf6" strokeWidth="1" strokeOpacity="0.4"
                  transform="rotate(60 14 14)" />
                <ellipse cx="14" cy="14" rx="13" ry="5" stroke="#10b981" strokeWidth="0.8" strokeOpacity="0.3"
                  transform="rotate(120 14 14)" />
              </svg>
            </div>
            <div>
              <span className="text-sm font-bold tracking-wide gradient-text">PandM</span>
              <span className="text-xs ml-1.5" style={{ color: '#334155' }}>量子知识图谱</span>
            </div>
          </div>

          {/* Separator */}
          <div className="w-px h-5 flex-shrink-0" style={{ background: 'rgba(255,255,255,0.08)' }} />

          {/* Breadcrumb */}
          <Breadcrumb
            ancestors={[getNode('root'), ...focusAncestors].filter((n, i, arr) =>
              n && arr.findIndex((x) => x?.id === n.id) === i
            )}
            currentNode={focusNode || getNode('root')}
            onNavigate={handleBreadcrumbNavigate}
          />
        </div>

        {/* Right: search + controls */}
        <div className="flex items-center gap-3 flex-shrink-0">
          <SearchBar
            query={searchQuery}
            onChange={setSearchQuery}
            resultCount={searchResultCount}
            total={graph.nodes.length}
          />

          {/* Reset focus button */}
          {focusNodeId && (
            <button
              onClick={() => setFocusNodeId(null)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-all"
              style={{ color: '#94a3b8', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.08)' }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.04)' }}
              title="返回全图"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l9-9 9 9M5 10v9a1 1 0 001 1h4v-5h4v5h4a1 1 0 001-1v-9" />
              </svg>
              全图
            </button>
          )}

          {/* Legend toggle */}
          <button
            onClick={() => setShowLegend((v) => !v)}
            className="p-2 rounded-lg transition-all"
            style={{
              color: showLegend ? '#00d4ff' : '#475569',
              background: showLegend ? 'rgba(0,212,255,0.1)' : 'rgba(255,255,255,0.04)',
              border: `1px solid ${showLegend ? 'rgba(0,212,255,0.3)' : 'rgba(255,255,255,0.08)'}`,
            }}
            title="节点图例"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
            </svg>
          </button>

          {/* Reset button */}
          <button
            onClick={handleResetGraph}
            className="p-2 rounded-lg transition-all"
            style={{ color: '#475569', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}
            title="重置图谱"
            onMouseEnter={(e) => { e.currentTarget.style.color = '#ef4444'; e.currentTarget.style.borderColor = 'rgba(239,68,68,0.3)' }}
            onMouseLeave={(e) => { e.currentTarget.style.color = '#475569'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)' }}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>
      </header>

      {/* Main graph area */}
      <div
        className="absolute inset-0"
        style={{ top: 56 }}
      >
        <QuantumGraph
          graph={graph}
          selectedNode={selectedNode}
          focusNodeId={focusNodeId}
          searchQuery={searchQuery}
          onSelectNode={handleSelectNode}
          onFocusChange={handleFocusChange}
        />
      </div>

      {/* Node panel (right sidebar) */}
      <div className="absolute top-14 bottom-0 right-0" style={{ width: 384, pointerEvents: selectedNode ? 'all' : 'none' }}>
        <AnimatePresence mode="wait">
          {selectedNode && (
            <NodePanel
              key={selectedNode.id}
              node={selectedNode}
              children={selectedChildren}
              ancestors={selectedAncestors}
              onClose={() => setSelectedNode(null)}
              onSelectNode={handleSelectNode}
              onAddNode={handleAddNode}
              onFocusChange={handleFocusChange}
              onDeleteNode={handleDeleteNode}
            />
          )}
        </AnimatePresence>
      </div>

      {/* Legend panel */}
      <AnimatePresence>
        {showLegend && <LegendPanel onClose={() => setShowLegend(false)} />}
      </AnimatePresence>

      {/* Add node modal */}
      <AnimatePresence>
        {addModal && (
          <AddNodeModal
            parentNode={getNode(addModal.parentId)}
            defaultType={addModal.defaultType}
            onConfirm={handleConfirmAdd}
            onCancel={() => setAddModal(null)}
          />
        )}
      </AnimatePresence>

      {/* Bottom status bar */}
      <div
        className="absolute bottom-0 left-0 right-0 flex items-center justify-between px-5 py-1.5 z-10"
        style={{
          background: 'rgba(7,7,15,0.8)',
          backdropFilter: 'blur(8px)',
          borderTop: '1px solid rgba(0,212,255,0.08)',
        }}
      >
        <div className="flex items-center gap-4 text-xs" style={{ color: '#334155' }}>
          <span>{graph.nodes.length} 节点</span>
          <span>{graph.edges.length} 连接</span>
          {searchQuery && (
            <span style={{ color: '#00d4ff' }}>
              搜索到 {searchResultCount} 个匹配
            </span>
          )}
        </div>
        <div className="text-xs flex items-center gap-3" style={{ color: '#334155' }}>
          <span>点击选中 · 双击聚焦 · 滚轮缩放 · 拖拽移动</span>
          <div className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: '#10b981' }} />
            <span style={{ color: '#10b981' }}>已连接本地存储</span>
          </div>
        </div>
      </div>

      {/* Floating add button (when no node selected) */}
      {!selectedNode && (
        <button
          onClick={() => setAddModal({ parentId: focusNodeId || 'root', defaultType: 'topic' })}
          className="absolute bottom-10 right-6 w-11 h-11 rounded-full flex items-center justify-center z-10 transition-all"
          style={{
            background: 'linear-gradient(135deg, rgba(0,212,255,0.2), rgba(139,92,246,0.2))',
            border: '1px solid rgba(0,212,255,0.35)',
            boxShadow: '0 0 20px rgba(0,212,255,0.2)',
            color: '#00d4ff',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.boxShadow = '0 0 30px rgba(0,212,255,0.4)'; e.currentTarget.style.transform = 'scale(1.1)' }}
          onMouseLeave={(e) => { e.currentTarget.style.boxShadow = '0 0 20px rgba(0,212,255,0.2)'; e.currentTarget.style.transform = 'scale(1)' }}
          title="新建节点"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
        </button>
      )}
    </div>
  )
}

function LegendPanel({ onClose }) {
  const items = [
    { type: 'root', label: '根节点', color: '#00d4ff', desc: '知识宇宙入口' },
    { type: 'module', label: '模块', color: '#8b5cf6', desc: '工作计划 / 学习资料' },
    { type: 'category', label: '分类', color: '#10b981', desc: '经典方法 / 量子计算' },
    { type: 'topic', label: '主题', color: '#f97316', desc: '具体知识点' },
    { type: 'leaf', label: '资料', color: '#f472b6', desc: '具体学习材料' },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      className="absolute top-16 right-16 z-30 rounded-xl overflow-hidden"
      style={{
        background: 'rgba(10,10,20,0.95)',
        border: '1px solid rgba(0,212,255,0.15)',
        boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        backdropFilter: 'blur(16px)',
        minWidth: 220,
      }}
    >
      <div className="px-4 py-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold" style={{ color: '#94a3b8' }}>节点类型图例</span>
          <button onClick={onClose} className="p-0.5 rounded hover:bg-white/10 transition-all" style={{ color: '#475569' }}>
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
      <div className="p-3 space-y-2">
        {items.map((item) => (
          <div key={item.type} className="flex items-center gap-3">
            <div className="relative flex-shrink-0 w-5 h-5">
              <div className="absolute inset-0 rounded-full opacity-30"
                style={{ background: item.color, filter: 'blur(4px)' }} />
              <div className="absolute inset-1 rounded-full"
                style={{ background: item.color, opacity: 0.85 }} />
            </div>
            <div>
              <div className="text-xs font-medium" style={{ color: '#e2e8f0' }}>{item.label}</div>
              <div className="text-xs" style={{ color: '#475569' }}>{item.desc}</div>
            </div>
          </div>
        ))}
        <div className="pt-2 mt-2 space-y-1.5" style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
          <div className="flex items-center gap-2 text-xs" style={{ color: '#475569' }}>
            <div className="w-6 h-px" style={{ background: 'rgba(0,212,255,0.5)' }} />
            <span>层级连接</span>
          </div>
          <div className="flex items-center gap-2 text-xs" style={{ color: '#475569' }}>
            <div className="w-6 h-px" style={{ background: 'rgba(139,92,246,0.5)', borderTop: '1px dashed rgba(139,92,246,0.5)' }} />
            <span>跨模块关联</span>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
