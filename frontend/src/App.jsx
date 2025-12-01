// frontend/src/App.jsx
import ImageUploader from './components/ImageUploader.jsx'; // 👈 Check this path

function App() {
  // Add any necessary wrapper div or styling here
  return (
    <div className="App">
      {/* This is the component you built that contains the 
        entire dashboard, form, and results logic.
      */}
      <ImageUploader /> 
    </div>
  );
}

export default App;