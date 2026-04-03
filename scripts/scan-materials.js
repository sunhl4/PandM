/**
 * scan-materials.js
 *
 * Walks the `materials/` directory tree and produces { nodes, edges } for the knowledge graph.
 *
 * RULES
 * ─────
 * • Each directory can carry a `.graphmeta.json` that controls how it maps to the graph:
 *     { "id": "existing-node-id" }           → directory IS an existing curated node; no new node created
 *     { "label": "...", "type": "topic" }    → create a new auto-disc node with this metadata
 *     { "excludeFiles": ["file.md", ...] }   → skip these files (already curated elsewhere)
 *     { "skip": true }                       → skip the whole directory
 *
 * • Files included: *.md, *.ipynb  (not *.QA.zh-CN.md, not README.md)
 * • Stable node IDs: "disc-" + sanitised relative path
 * • Top-level dirs (learning/, software/, work-plan/) attach directly to existing module nodes
 */

import fs   from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname  = path.dirname(fileURLToPath(import.meta.url))
export const ROOT = path.resolve(__dirname, '..')
const MATERIALS  = path.join(ROOT, 'materials')

const GITHUB_RAW  = 'https://github.com/sunhl4/PandM/blob/main/materials'
const NBVIEWER    = 'https://nbviewer.org/github/sunhl4/PandM/blob/main/materials'

const SKIP_DIRS   = new Set(['node_modules', '__pycache__', '.git', '.pytest_cache',
                              'tests', '.github', '.venv', 'dist', 'build'])
const CONTENT_EXT = new Set(['.md', '.ipynb'])

/** Read .graphmeta.json from a directory (returns {} on missing/error) */
function readMeta(absDir) {
  try {
    const p = path.join(absDir, '.graphmeta.json')
    if (fs.existsSync(p)) return JSON.parse(fs.readFileSync(p, 'utf8'))
  } catch {}
  return {}
}

/** Generate a stable, deterministic node id from a relative path */
function pathToId(relPath) {
  return 'disc-' + relPath
    .replace(/\\/g, '/')
    .replace(/\.(md|ipynb)$/, '')
    .replace(/\//g, '-')
    .replace(/[^a-zA-Z0-9-]/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .toLowerCase()
    .slice(0, 64)
}

/** Extract the first # heading from a .md or first markdown cell of a .ipynb */
function extractTitle(filePath) {
  const ext = path.extname(filePath).toLowerCase()
  try {
    if (ext === '.md') {
      const txt = fs.readFileSync(filePath, 'utf8').slice(0, 3000)
      const m = txt.match(/^#\s+(.+)/m)
      if (m) return m[1].trim()
    }
    if (ext === '.ipynb') {
      const nb = JSON.parse(fs.readFileSync(filePath, 'utf8'))
      for (const cell of (nb.cells || [])) {
        if (cell.cell_type === 'markdown') {
          const src = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '')
          const m = src.match(/^#\s+(.+)/m)
          if (m) return m[1].trim()
        }
      }
    }
  } catch {}
  const base = path.basename(filePath, path.extname(filePath))
  return base.replace(/[_-]/g, ' ').replace(/^\d+\s+/, '').trim() || base
}

/** Extract first non-header, non-table paragraph (≤120 chars) */
function extractDesc(filePath) {
  try {
    const ext = path.extname(filePath).toLowerCase()
    if (ext !== '.md') return ''
    const lines = fs.readFileSync(filePath, 'utf8').slice(0, 4000).split('\n')
    let seenHeading = false
    for (const line of lines) {
      if (line.startsWith('#')) { seenHeading = true; continue }
      if (!seenHeading) continue
      const t = line.trim()
      if (t && !t.startsWith('#') && !t.startsWith('|') && !t.startsWith('!') &&
          !t.startsWith('-') && !t.startsWith('```') && !t.startsWith('*')) {
        return t.slice(0, 120)
      }
    }
  } catch {}
  return ''
}

/** Core recursive scanner */
function walkDir(absDir, relPath, parentNodeId, depth, nodes, edges) {
  if (depth > 7) return

  const meta       = readMeta(absDir)
  if (meta.skip)   return

  const excludeSet = new Set(meta.excludeFiles || [])
  let entries
  try { entries = fs.readdirSync(absDir, { withFileTypes: true }) } catch { return }

  // ── Determine this directory's effective node id ──────────────────────────
  let thisDirNodeId

  if (meta.id) {
    // This directory already has a curated node — just use it as the attachment point
    thisDirNodeId = meta.id
  } else {
    // Auto-create a new node for this directory
    const dirName = path.basename(absDir)
    thisDirNodeId = pathToId(relPath)

    let desc = meta.description || ''
    const readmePath = path.join(absDir, 'README.md')
    if (!desc && fs.existsSync(readmePath)) desc = extractDesc(readmePath)

    const nodeType = depth <= 1 ? 'category' : 'topic'
    nodes.push({
      id:            thisDirNodeId,
      label:         (meta.label || dirName.replace(/[_-]/g, ' ')).slice(0, 52),
      type:          meta.type || nodeType,
      description:   (desc || `${dirName} folder`).slice(0, 120),
      content:       `# ${meta.label || dirName}\n\nAuto-discovered from \`materials/${relPath}/\`\n\n${desc}`,
      tags:          ['auto-discovered', ...(meta.tags || [])],
      links:         [],
      _autoDiscovered: true,
      _sourcePath:   relPath + '/',
    })
    if (parentNodeId) edges.push({ source: parentNodeId, target: thisDirNodeId })
  }

  // ── Process children ──────────────────────────────────────────────────────
  for (const entry of entries) {
    if (SKIP_DIRS.has(entry.name)) continue
    const fullPath = path.join(absDir, entry.name)
    const rel      = relPath ? `${relPath}/${entry.name}` : entry.name

    if (entry.isDirectory()) {
      walkDir(fullPath, rel, thisDirNodeId, depth + 1, nodes, edges)
    } else if (entry.isFile()) {
      const ext = path.extname(entry.name).toLowerCase()
      if (!CONTENT_EXT.has(ext))                        continue
      if (excludeSet.has(entry.name))                    continue
      if (entry.name.endsWith('.QA.zh-CN.md'))           continue
      if (entry.name === 'README.md')                    continue
      if (entry.name === '.graphmeta.json')              continue

      const nodeId = pathToId(rel)
      const title  = extractTitle(fullPath)
      const desc   = extractDesc(fullPath)
      const isNb   = ext === '.ipynb'
      const url    = `${isNb ? NBVIEWER : GITHUB_RAW}/${rel.replace(/\\/g, '/')}`
      const isZhCN = entry.name.endsWith('.zh-CN.md')
      const fileTag = isNb ? 'notebook' : isZhCN ? 'translation' : 'markdown'

      nodes.push({
        id:            nodeId,
        label:         title.slice(0, 52),
        type:          'leaf',
        description:   (desc || entry.name).slice(0, 120),
        content: [
          `# ${title}`,
          '',
          `**File**: \`${entry.name}\``,
          desc ? `\n${desc}` : '',
          '',
          `Path: \`materials/${rel}\``,
        ].join('\n'),
        tags:          ['auto-discovered', fileTag],
        links:         [{ label: isNb ? 'Open in nbviewer' : 'View file', url }],
        _autoDiscovered: true,
        _sourcePath:   rel,
      })
      if (thisDirNodeId) edges.push({ source: thisDirNodeId, target: nodeId })
    }
  }
}

// ── Top-level module mappings ─────────────────────────────────────────────────
const TOP_MODULE_MAP = {
  'learning':  'learning',
  'software':  'software',
  'work-plan': 'work-plan',
}

/**
 * Main entry point.
 * Returns { nodes: Node[], edges: Edge[] } — ONLY auto-discovered items.
 * Merge with INITIAL_GRAPH in graphData.js.
 */
export function scan(materialsDir = MATERIALS) {
  const nodes = []
  const edges = []

  for (const [dirName, moduleNodeId] of Object.entries(TOP_MODULE_MAP)) {
    const absTop = path.join(materialsDir, dirName)
    if (!fs.existsSync(absTop)) continue

    // Scan files directly inside top-level dir
    const topEntries = fs.readdirSync(absTop, { withFileTypes: true })
    for (const e of topEntries) {
      const fullPath = path.join(absTop, e.name)
      const rel      = `${dirName}/${e.name}`

      if (e.isDirectory()) {
        if (SKIP_DIRS.has(e.name)) continue
        walkDir(fullPath, rel, moduleNodeId, 1, nodes, edges)
      } else if (e.isFile()) {
        const ext = path.extname(e.name).toLowerCase()
        if (!CONTENT_EXT.has(ext)) continue
        if (e.name === 'README.md') continue
        if (e.name.endsWith('.QA.zh-CN.md')) continue

        const nodeId = pathToId(rel)
        const title  = extractTitle(fullPath)
        const desc   = extractDesc(fullPath)
        const isNb   = ext === '.ipynb'
        const url    = `${isNb ? NBVIEWER : GITHUB_RAW}/${rel}`
        nodes.push({
          id:            nodeId,
          label:         title.slice(0, 52),
          type:          'leaf',
          description:   (desc || e.name).slice(0, 120),
          content:       `# ${title}\n\n**File**: \`${e.name}\`\n\n${desc}\n\nPath: \`materials/${rel}\``,
          tags:          ['auto-discovered', isNb ? 'notebook' : 'markdown'],
          links:         [{ label: isNb ? 'Open in nbviewer' : 'View file', url }],
          _autoDiscovered: true,
          _sourcePath:   rel,
        })
        edges.push({ source: moduleNodeId, target: nodeId })
      }
    }
  }

  return { nodes, edges }
}
