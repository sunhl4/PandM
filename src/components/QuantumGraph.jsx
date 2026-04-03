import { useEffect, useRef, useCallback } from 'react'
import * as d3 from 'd3'
import { NODE_TYPE_CONFIG } from '../data/initialGraph.js'

const TYPE_ORDER = ['root', 'module', 'category', 'topic', 'leaf']

function getTypeConfig(type) {
  return NODE_TYPE_CONFIG[type] || NODE_TYPE_CONFIG.leaf
}

export default function QuantumGraph({
  graph,
  selectedNode,
  focusNodeId,
  searchQuery,
  onSelectNode,
  onFocusChange,
}) {
  const svgRef = useRef(null)
  const simRef = useRef(null)
  const zoomRef = useRef(null)
  const gRef = useRef(null)

  // Build visible graph: show all nodes but dim non-focused subtree
  const buildVisibleGraph = useCallback((graph, focusId) => {
    if (!focusId || focusId === 'root') return graph

    // Collect subtree rooted at focusId
    const childrenMap = {}
    graph.edges.forEach((e) => {
      const src = typeof e.source === 'object' ? e.source.id : e.source
      const tgt = typeof e.target === 'object' ? e.target.id : e.target
      if (!childrenMap[src]) childrenMap[src] = []
      childrenMap[src].push(tgt)
    })
    const parentMap = {}
    graph.edges.forEach((e) => {
      const src = typeof e.source === 'object' ? e.source.id : e.source
      const tgt = typeof e.target === 'object' ? e.target.id : e.target
      if (!e.type) parentMap[tgt] = src
    })

    const visible = new Set()
    // Add ancestors
    let cur = focusId
    while (cur) { visible.add(cur); cur = parentMap[cur] }
    // Add descendants
    const queue = [focusId]
    while (queue.length) {
      const id = queue.shift()
      visible.add(id)
      ;(childrenMap[id] || []).forEach((c) => queue.push(c))
    }

    return {
      nodes: graph.nodes.filter((n) => visible.has(n.id)),
      edges: graph.edges.filter((e) => {
        const src = typeof e.source === 'object' ? e.source.id : e.source
        const tgt = typeof e.target === 'object' ? e.target.id : e.target
        return visible.has(src) && visible.has(tgt)
      }),
    }
  }, [])

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    const container = svgRef.current.parentElement
    let W = container.clientWidth
    let H = container.clientHeight

    svg.attr('width', W).attr('height', H)

    // Setup defs for filters & gradients
    svg.selectAll('defs').remove()
    const defs = svg.append('defs')

    // Glow filter
    Object.entries(NODE_TYPE_CONFIG).forEach(([type, cfg]) => {
      const filter = defs.append('filter').attr('id', `glow-${type}`).attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%')
      filter.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'blur')
      const merge = filter.append('feMerge')
      merge.append('feMergeNode').attr('in', 'blur')
      merge.append('feMergeNode').attr('in', 'SourceGraphic')
    })

    // Arrow marker for directed edges
    defs.append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -4 8 8')
      .attr('refX', 28)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-4L8,0L0,4')
      .attr('fill', 'rgba(0,212,255,0.5)')

    defs.append('marker')
      .attr('id', 'arrow-cross')
      .attr('viewBox', '0 -4 8 8')
      .attr('refX', 28)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-4L8,0L0,4')
      .attr('fill', 'rgba(139,92,246,0.5)')

    // Main group with zoom
    svg.selectAll('g.root-group').remove()
    const g = svg.append('g').attr('class', 'root-group')
    gRef.current = g

    const zoom = d3.zoom()
      .scaleExtent([0.15, 4])
      .filter((event) => {
        const onNode = event.target?.closest?.('g.node')
        if (onNode && (event.type === 'mousedown' || event.type === 'touchstart' || event.type === 'dblclick')) {
          return false
        }
        return (!event.ctrlKey || event.type === 'wheel') && !event.button
      })
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })
    zoomRef.current = zoom
    svg.call(zoom)

    // Initial zoom to center
    svg.call(zoom.transform, d3.zoomIdentity.translate(W / 2, H / 2).scale(0.7))

    const renderGraph = (visGraph) => {
      g.selectAll('*').remove()

      const nodes = visGraph.nodes.map((n) => ({ ...n }))
      const edges = visGraph.edges.map((e) => ({
        ...e,
        source: typeof e.source === 'object' ? e.source.id : e.source,
        target: typeof e.target === 'object' ? e.target.id : e.target,
      }))

      // Level-based x positioning hint
      const levelMap = {}
      const typeIdx = {}
      TYPE_ORDER.forEach((t, i) => { typeIdx[t] = i })
      nodes.forEach((n) => { levelMap[n.id] = typeIdx[n.type] || 2 })

      // Force simulation
      if (simRef.current) simRef.current.stop()
      const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(edges).id((d) => d.id).distance((e) => {
          const srcType = nodes.find((n) => n.id === (typeof e.source === 'object' ? e.source.id : e.source))?.type
          const dist = { root: 280, module: 215, category: 170, topic: 125, leaf: 95 }
          return dist[srcType] || 155
        }).strength((e) => (e.type === 'cross' ? 0.03 : 0.12)))
        .force('charge', d3.forceManyBody().strength((d) => {
          const s = { root: -520, module: -340, category: -220, topic: -130, leaf: -70 }
          return s[d.type] || -130
        }))
        .force('center', d3.forceCenter(0, 0))
        .force('collision', d3.forceCollide().radius((d) => getTypeConfig(d.type).radius + 36).strength(0.72))
        .velocityDecay(0.35)
        .alphaDecay(0.022)
      simRef.current = simulation

      // Draw edges
      const linkGroup = g.append('g').attr('class', 'links')
      const link = linkGroup.selectAll('path')
        .data(edges)
        .join('path')
        .attr('fill', 'none')
        .attr('stroke', (d) => d.type === 'cross' ? 'rgba(139,92,246,0.35)' : 'rgba(0,212,255,0.3)')
        .attr('stroke-width', (d) => d.type === 'cross' ? 1 : 1.5)
        .attr('stroke-dasharray', (d) => d.type === 'cross' ? '5,4' : '0')
        .attr('marker-end', (d) => d.type === 'cross' ? 'url(#arrow-cross)' : 'url(#arrow)')

      // Animated flowing particles on links
      const linkParticles = linkGroup.selectAll('circle.link-particle')
        .data(edges.filter((e) => e.type !== 'cross'))
        .join('circle')
        .attr('class', 'link-particle')
        .attr('r', 2)
        .attr('fill', 'rgba(0,212,255,0.7)')

      // Draw node groups
      const nodeGroup = g.append('g').attr('class', 'nodes')
      const nodeEl = nodeGroup.selectAll('g.node')
        .data(nodes)
        .join('g')
        .attr('class', 'node')
        .style('cursor', 'pointer')

      // Outer glow ring (animated for selected)
      nodeEl.each(function (d) {
        const el = d3.select(this)
        const cfg = getTypeConfig(d.type)

        // Orbital ring for module+ nodes
        if (d.type === 'root' || d.type === 'module') {
          el.append('ellipse')
            .attr('rx', cfg.radius + 14)
            .attr('ry', (cfg.radius + 14) * 0.38)
            .attr('fill', 'none')
            .attr('stroke', cfg.color)
            .attr('stroke-width', 0.8)
            .attr('stroke-opacity', 0.35)
            .attr('class', 'orbit-ring-1')
          el.append('ellipse')
            .attr('rx', cfg.radius + 20)
            .attr('ry', (cfg.radius + 20) * 0.28)
            .attr('fill', 'none')
            .attr('stroke', cfg.color)
            .attr('stroke-width', 0.6)
            .attr('stroke-opacity', 0.2)
            .attr('class', 'orbit-ring-2')
            .attr('transform', 'rotate(60)')
        }

        // Outer glow circle
        el.append('circle')
          .attr('r', cfg.radius + 8)
          .attr('fill', 'none')
          .attr('stroke', cfg.color)
          .attr('stroke-width', 1.2)
          .attr('stroke-opacity', 0.2)
          .attr('class', 'glow-ring')
          .attr('filter', `url(#glow-${d.type})`)

        // Main circle
        el.append('circle')
          .attr('r', cfg.radius)
          .attr('fill', `radial-gradient(circle, ${cfg.color}22, ${cfg.color}08)`)
          .attr('stroke', cfg.color)
          .attr('stroke-width', 1.8)
          .attr('class', 'main-circle')
          .attr('filter', `url(#glow-${d.type})`)
          .style('fill', () => {
            // Use inline SVG gradient approximation
            return `rgba(${hexToRgb(cfg.color)},0.12)`
          })

        // Inner core dot
        el.append('circle')
          .attr('r', Math.max(cfg.radius * 0.28, 3))
          .attr('fill', cfg.color)
          .attr('fill-opacity', 0.85)
          .attr('class', 'core-dot')

        // Label
        const fontSize = { root: 13, module: 12, category: 11, topic: 10, leaf: 9 }[d.type] || 10
        el.append('text')
          .attr('dy', cfg.radius + 14)
          .attr('text-anchor', 'middle')
          .attr('fill', '#e2e8f0')
          .attr('font-size', fontSize)
          .attr('font-family', 'Inter, sans-serif')
          .attr('font-weight', d.type === 'root' || d.type === 'module' ? '600' : '400')
          .attr('pointer-events', 'none')
          .text(d.label)

        // Hover zone (transparent larger circle)
        el.append('circle')
          .attr('r', cfg.radius + 10)
          .attr('fill', 'transparent')
          .attr('class', 'hit-zone')
      })

      // Drag behavior
      const drag = d3.drag()
        .on('start', (event, d) => {
          event.sourceEvent?.stopPropagation?.()
          if (!event.active) simulation.alphaTarget(0.55).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on('drag', (event, d) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0)
          d.fx = null
          d.fy = null
        })
      nodeEl.call(drag)

      // Click handler
      nodeEl.on('click', (event, d) => {
        event.stopPropagation()
        onSelectNode(d)
      })

      // Double-click → focus on this node's subtree
      nodeEl.on('dblclick', (event, d) => {
        event.stopPropagation()
        onFocusChange(d.id)
      })

      // Hover effects
      nodeEl.on('mouseenter', function (event, d) {
        const cfg = getTypeConfig(d.type)
        d3.select(this).select('.glow-ring')
          .transition().duration(200)
          .attr('stroke-opacity', 0.7)
          .attr('r', cfg.radius + 12)
        d3.select(this).select('.main-circle')
          .transition().duration(200)
          .attr('r', cfg.radius * 1.1)
        d3.select(this).select('.core-dot')
          .transition().duration(200)
          .attr('r', Math.max(cfg.radius * 0.35, 4))
      })
      nodeEl.on('mouseleave', function (event, d) {
        const cfg = getTypeConfig(d.type)
        d3.select(this).select('.glow-ring')
          .transition().duration(300)
          .attr('stroke-opacity', 0.2)
          .attr('r', cfg.radius + 8)
        d3.select(this).select('.main-circle')
          .transition().duration(300)
          .attr('r', cfg.radius)
        d3.select(this).select('.core-dot')
          .transition().duration(300)
          .attr('r', Math.max(cfg.radius * 0.28, 3))
      })

      // Tick
      let t = 0
      simulation.on('tick', () => {
        t++
        link.attr('d', (d) => {
          const sx = d.source.x || 0, sy = d.source.y || 0
          const tx = d.target.x || 0, ty = d.target.y || 0
          const dx = tx - sx, dy = ty - sy
          const dr = Math.sqrt(dx * dx + dy * dy) * 0.5
          return `M${sx},${sy}A${dr},${dr},0,0,1,${tx},${ty}`
        })

        // Flowing particles along links
        linkParticles.attr('cx', (d) => {
          const progress = (t * 0.008 + d.index * 0.3) % 1
          return (d.source.x || 0) + ((d.target.x || 0) - (d.source.x || 0)) * progress
        }).attr('cy', (d) => {
          const progress = (t * 0.008 + d.index * 0.3) % 1
          return (d.source.y || 0) + ((d.target.y || 0) - (d.source.y || 0)) * progress
        }).attr('opacity', (d) => {
          const progress = (t * 0.008 + d.index * 0.3) % 1
          return Math.sin(progress * Math.PI) * 0.8
        })

        nodeEl.attr('transform', (d) => `translate(${d.x || 0},${d.y || 0})`)

        // Animate orbital rings
        nodeEl.selectAll('.orbit-ring-1')
          .attr('transform', `rotate(${t * 0.3})`)
        nodeEl.selectAll('.orbit-ring-2')
          .attr('transform', `rotate(${-t * 0.2 + 60})`)
      })

      return simulation
    }

    const visGraph = buildVisibleGraph(graph, focusNodeId)
    renderGraph(visGraph)

    // Resize handler
    const handleResize = () => {
      W = container.clientWidth
      H = container.clientHeight
      svg.attr('width', W).attr('height', H)
      if (simRef.current) {
        simRef.current.force('center', d3.forceCenter(0, 0))
        simRef.current.alpha(0.3).restart()
      }
    }
    window.addEventListener('resize', handleResize)
    return () => {
      if (simRef.current) simRef.current.stop()
      window.removeEventListener('resize', handleResize)
      svg.selectAll('*').remove()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graph, focusNodeId, buildVisibleGraph])

  // Apply search highlight as a separate effect (no re-simulation)
  useEffect(() => {
    if (!gRef.current) return
    const q = searchQuery?.toLowerCase().trim()
    gRef.current.selectAll('g.node').each(function (d) {
      const el = d3.select(this)
      if (!q) {
        el.style('opacity', 1)
        el.select('.main-circle').attr('stroke-width', 1.8)
        return
      }
      const match = d.label.toLowerCase().includes(q) ||
        (d.tags || []).some((t) => t.toLowerCase().includes(q)) ||
        (d.description || '').toLowerCase().includes(q)
      if (match) {
        el.style('opacity', 1)
        el.select('.main-circle').attr('stroke-width', 3)
        el.select('.glow-ring').attr('stroke-opacity', 0.9)
      } else {
        el.style('opacity', 0.15)
        el.select('.main-circle').attr('stroke-width', 1.8)
      }
    })
  }, [searchQuery])

  // Highlight selected node
  useEffect(() => {
    if (!gRef.current) return
    gRef.current.selectAll('g.node').each(function (d) {
      const el = d3.select(this)
      const cfg = getTypeConfig(d.type)
      if (d.id === selectedNode?.id) {
        el.select('.main-circle')
          .attr('stroke-width', 3)
          .style('fill', `rgba(${hexToRgb(cfg.color)},0.25)`)
        el.select('.glow-ring').attr('stroke-opacity', 0.8).attr('r', cfg.radius + 14)
      } else {
        el.select('.main-circle')
          .attr('stroke-width', 1.8)
          .style('fill', `rgba(${hexToRgb(cfg.color)},0.12)`)
        el.select('.glow-ring').attr('stroke-opacity', 0.2).attr('r', cfg.radius + 8)
      }
    })
  }, [selectedNode])

  return (
    <svg
      ref={svgRef}
      className="w-full h-full"
      style={{ background: 'transparent' }}
      onClick={() => onSelectNode(null)}
    />
  )
}

function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
  return result
    ? `${parseInt(result[1], 16)},${parseInt(result[2], 16)},${parseInt(result[3], 16)}`
    : '0,212,255'
}
