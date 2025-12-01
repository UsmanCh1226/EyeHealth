import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // This is the key part! It sets up the proxy.
    proxy: {
      // If the frontend sees a request for this URL path...
      '/predict_eye_health': {
        // ...it redirects it to the backend server running on port 5000 (Flask)
        target: 'http://127.0.0.1:5000', 
        changeOrigin: true, // Necessary for many local development setups
        secure: false, // For non-HTTPS development
      },
      // You may need to add this for the Gemini report if you create a separate endpoint:
      '/generate_report': {
        target: 'http://127.0.0.1:5000', 
        changeOrigin: true, 
        secure: false,
      },
    }
  }
})