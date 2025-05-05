import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  base: '/static',
  build: {
    outDir: '../frontend/dist',
    assetsDir: 'assets',
    emptyOutDir: true
  }
})