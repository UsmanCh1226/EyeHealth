// frontend/src/main.jsx - Minimal and Correct
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';

// If you have a global CSS file, uncomment the line below.
// import './index.css'; 

// This ensures React correctly starts the application on the 'root' element
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);