import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3002,
    proxy: {
      '/data-profiling': 'http://localhost:8002',
      '/data-quality': 'http://localhost:8002',
      '/ai-analysis': 'http://localhost:8002',
      '/api': 'http://localhost:8002',
      '/health': 'http://localhost:8002'
    }
  }
})
