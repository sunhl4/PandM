import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { scanGraphPlugin } from './vite-plugin-scan.js'

// GitHub Pages 部署在 /PandM/；本地 dev 用根路径，避免打开 localhost:5173/ 空白
export default defineConfig(({ command }) => ({
  plugins: [react(), scanGraphPlugin()],
  base: command === 'build' ? '/PandM/' : '/',
  server: {
    port: 5173,
    open: true,
  },
}))
