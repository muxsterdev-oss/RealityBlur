import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
    proxy: {
      // Proxy API requests to the backend during development
      '/api': {
        // Use 127.0.0.1 to avoid IPv6 ::1 resolution issues in some environments
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false,
        ws: false
      }
    }
  }
})
