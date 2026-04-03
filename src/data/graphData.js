/**
 * graphData.js
 *
 * Merges the curated INITIAL_GRAPH with auto-discovered SCANNED_GRAPH.
 *
 * Priority rules
 * ──────────────
 * • Curated nodes (INITIAL_GRAPH) always win when IDs collide.
 * • Auto-discovered nodes are appended only if their id doesn't already exist.
 * • Same deduplication logic for edges.
 *
 * useLocalStorage already re-merges on every app load:
 *   "keep stored nodes + add any new nodes from BASE_GRAPH that aren't stored yet"
 * So adding new .md files will automatically appear in the graph on the next page load.
 */

import { INITIAL_GRAPH }  from './initialGraph.js'
import { SCANNED_GRAPH }  from 'virtual:scanned-graph'

function mergeGraphs(curated, scanned) {
  const curatedIds   = new Set(curated.nodes.map((n) => n.id))
  const curatedEdges = new Set(curated.edges.map((e) => `${e.source}||${e.target}`))

  const newNodes = scanned.nodes.filter((n) => !curatedIds.has(n.id))
  const newEdges = scanned.edges.filter((e) => {
    const key = `${e.source}||${e.target}`
    if (curatedEdges.has(key)) return false
    // Only add edge if both endpoints will exist in the merged graph
    const tgt = e.target
    const src = e.source
    const srcExists = curatedIds.has(src) || newNodes.some((n) => n.id === src)
    const tgtExists = curatedIds.has(tgt) || newNodes.some((n) => n.id === tgt)
    return srcExists && tgtExists
  })

  return {
    nodes: [...curated.nodes, ...newNodes],
    edges: [...curated.edges, ...newEdges],
  }
}

export const BASE_GRAPH = mergeGraphs(INITIAL_GRAPH, SCANNED_GRAPH)
