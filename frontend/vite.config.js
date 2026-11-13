import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // in development, send /api requests to the FastAPI server, so the
    // browser sees one address (no CORS setup needed)
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
  test: {
    environment: 'jsdom', // a simulated browser for the tests
    setupFiles: './src/test/setup.js',
  },
})
