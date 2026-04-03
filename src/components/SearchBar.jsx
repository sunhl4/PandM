import { useState, useRef, useEffect } from 'react'

export default function SearchBar({ query, onChange, resultCount, total }) {
  const [focused, setFocused] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault()
        inputRef.current?.focus()
      }
      if (e.key === 'Escape') {
        onChange('')
        inputRef.current?.blur()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onChange])

  return (
    <div className="relative flex items-center">
      <div
        className="flex items-center gap-2.5 px-3 py-2 rounded-xl transition-all"
        style={{
          background: focused ? 'rgba(0,212,255,0.08)' : 'rgba(255,255,255,0.04)',
          border: `1px solid ${focused ? 'rgba(0,212,255,0.4)' : 'rgba(255,255,255,0.1)'}`,
          boxShadow: focused ? '0 0 16px rgba(0,212,255,0.15)' : 'none',
          minWidth: 240,
          transition: 'all 0.2s ease',
        }}
      >
        <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"
          style={{ color: focused ? '#00d4ff' : '#64748b' }}>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder="搜索节点..."
          className="bg-transparent outline-none text-sm w-full"
          style={{ color: '#e2e8f0', caretColor: '#00d4ff' }}
        />
        {query && (
          <div className="flex items-center gap-2 flex-shrink-0">
            <span className="text-xs" style={{ color: '#00d4ff' }}>
              {resultCount}/{total}
            </span>
            <button
              onClick={() => onChange('')}
              className="p-0.5 rounded hover:bg-white/10 transition-all"
              style={{ color: '#64748b' }}
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}
        {!query && (
          <kbd className="hidden sm:flex items-center gap-0.5 text-xs px-1.5 py-0.5 rounded flex-shrink-0"
            style={{ background: 'rgba(255,255,255,0.06)', color: '#475569', fontFamily: 'monospace' }}>
            ⌘K
          </kbd>
        )}
      </div>
    </div>
  )
}
