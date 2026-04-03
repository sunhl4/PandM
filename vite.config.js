import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { scanGraphPlugin } from './vite-plugin-scan.js'

export default defineConfig({
  plugins: [react(), scanGraphPlugin()],
  base: '/PandM/',
  server: {
    port: 5173,
    open: true,
  },
})
