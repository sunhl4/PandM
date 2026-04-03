import { useCallback, useMemo } from 'react'

export function useGraph(graph, setGraph) {
  // Build a parent-children map
  const childrenMap = useMemo(() => {
    const map = {}
    graph.nodes.forEach((n) => { map[n.id] = [] })
    graph.edges.forEach((e) => {
      const src = typeof e.source === 'object' ? e.source.id : e.source
      const tgt = typeof e.target === 'object' ? e.target.id : e.target
      if (!map[src]) map[src] = []
      if (!map[src].includes(tgt)) map[src].push(tgt)
    })
    return map
  }, [graph])

  const parentMap = useMemo(() => {
    const map = {}
    graph.edges.forEach((e) => {
      const src = typeof e.source === 'object' ? e.source.id : e.source
      const tgt = typeof e.target === 'object' ? e.target.id : e.target
      if (e.type !== 'cross') map[tgt] = src
    })
    return map
  }, [graph])

  const getAncestors = useCallback((nodeId) => {
    const ancestors = []
    let current = nodeId
    while (parentMap[current]) {
      ancestors.unshift(parentMap[current])
      current = parentMap[current]
    }
    return ancestors
  }, [parentMap])

  const getNode = useCallback((id) => graph.nodes.find((n) => n.id === id), [graph])

  const getChildren = useCallback((id) => {
    return (childrenMap[id] || []).map((cid) => graph.nodes.find((n) => n.id === cid)).filter(Boolean)
  }, [childrenMap, graph])

  const addNode = useCallback((parentId, nodeData) => {
    const id = `node-${Date.now()}`
    const newNode = {
      id,
      label: nodeData.label,
      type: nodeData.type || 'topic',
      description: nodeData.description || '',
      content: nodeData.content || `# ${nodeData.label}\n\n${nodeData.description || ''}`,
      tags: nodeData.tags || [],
      links: nodeData.links || [],
    }
    const newEdge = { source: parentId, target: id }
    setGraph((prev) => ({
      nodes: [...prev.nodes, newNode],
      edges: [...prev.edges, newEdge],
    }))
    return id
  }, [setGraph])

  const updateNode = useCallback((id, updates) => {
    setGraph((prev) => ({
      ...prev,
      nodes: prev.nodes.map((n) => n.id === id ? { ...n, ...updates } : n),
    }))
  }, [setGraph])

  const deleteNode = useCallback((id) => {
    // Recursively collect all descendant ids
    const toDelete = new Set()
    const queue = [id]
    while (queue.length) {
      const cur = queue.shift()
      toDelete.add(cur)
      ;(childrenMap[cur] || []).forEach((c) => queue.push(c))
    }
    setGraph((prev) => ({
      nodes: prev.nodes.filter((n) => !toDelete.has(n.id)),
      edges: prev.edges.filter((e) => {
        const src = typeof e.source === 'object' ? e.source.id : e.source
        const tgt = typeof e.target === 'object' ? e.target.id : e.target
        return !toDelete.has(src) && !toDelete.has(tgt)
      }),
    }))
  }, [childrenMap, setGraph])

  return { getNode, getChildren, getAncestors, addNode, updateNode, deleteNode, childrenMap, parentMap }
}
