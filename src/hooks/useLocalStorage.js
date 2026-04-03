import { useState, useCallback } from 'react'

const STORAGE_KEY = 'pandm-graph-v1'

export function useLocalStorage(initialValue) {
  const [stored, setStored] = useState(() => {
    try {
      const item = window.localStorage.getItem(STORAGE_KEY)
      if (!item) return initialValue
      const parsed = JSON.parse(item)
      // Merge: keep all initial nodes that don't exist in stored
      const storedIds = new Set(parsed.nodes.map((n) => n.id))
      const newNodes = initialValue.nodes.filter((n) => !storedIds.has(n.id))
      const storedEdgeKeys = new Set(
        parsed.edges.map((e) => `${e.source}-${e.target}`)
      )
      const newEdges = initialValue.edges.filter(
        (e) => !storedEdgeKeys.has(`${e.source}-${e.target}`)
      )
      return {
        nodes: [...parsed.nodes, ...newNodes],
        edges: [...parsed.edges, ...newEdges],
      }
    } catch {
      return initialValue
    }
  })

  const setValue = useCallback((value) => {
    try {
      const toStore = typeof value === 'function' ? value(stored) : value
      setStored(toStore)
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(toStore))
    } catch (e) {
      console.warn('localStorage write failed', e)
    }
  }, [stored])

  const reset = useCallback(() => {
    window.localStorage.removeItem(STORAGE_KEY)
    setStored(initialValue)
  }, [initialValue])

  return [stored, setValue, reset]
}
