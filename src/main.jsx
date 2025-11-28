import React from 'react';
import { createRoot } from 'react-dom/client';
// FIX: Explicitly adding the .jsx extension back to ensure correct resolution
import App from './dashboard/web/App.jsx';

// Find the root element in index.html
const container = document.getElementById('root');
const root = createRoot(container);

// Render the App component into the root
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);