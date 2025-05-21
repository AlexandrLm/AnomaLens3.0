import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8001/api', // URL вашего бэкенд-сервера
        changeOrigin: true, // необходимо для виртуальных хостов
        rewrite: (path) => path.replace(/^\/api/, ''), // убираем /api из пути запроса
      },
    },
  },
})
