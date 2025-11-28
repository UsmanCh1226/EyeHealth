import React, { useState, useEffect, useCallback } from 'react';
import { initializeApp } from 'firebase/app';
import { getAuth, signInAnonymously, signInWithCustomToken, onAuthStateChanged } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore'; // Included for required setup, though not used for this specific demo

// --- Global Variables (Mandatory for Canvas Environment) ---
// These variables are provided by the hosting environment.
const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';
const firebaseConfig = typeof __firebase_config !== 'undefined' ? JSON.parse(__firebase_config) : {};
const initialAuthToken = typeof __initial_auth_token !== 'undefined' ? __initial_auth_token : null;

// The main application component
const App = () => {
  const [imageFile, setImageFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isAuthReady, setIsAuthReady] = useState(false);
  const [userId, setUserId] = useState(null);

  // Initialize Firebase and handle authentication
  useEffect(() => {
    if (!firebaseConfig || Object.keys(firebaseConfig).length === 0) {
        console.error("Firebase configuration is missing.");
        setIsAuthReady(true); // Allow UI to render even if DB fails
        return;
    }

    const app = initializeApp(firebaseConfig);
    const auth = getAuth(app);
    // const db = getFirestore(app); // Firestore initialization

    const setupAuth = async () => {
      try {
        if (initialAuthToken) {
          await signInWithCustomToken(auth, initialAuthToken);
        } else {
          await signInAnonymously(auth);
        }
      } catch (e) {
        console.error("Firebase Auth Error:", e);
      }
    };

    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        setUserId(user.uid);
      } else {
        setUserId(crypto.randomUUID()); // Anonymous or new session user
      }
      setIsAuthReady(true);
    });

    setupAuth();
    return () => unsubscribe();
  }, []);


  // Utility function to convert File/Blob to Base64 string
  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(',')[1]); // Only the data part
      reader.onerror = (error) => reject(error);
      reader.readAsDataURL(file);
    });
  };

  // The core function to call the Gemini Vision API
  const analyzeRetinaScan = useCallback(async (base64Image, mimeType) => {
    if (!isAuthReady) {
      setError("Authentication not ready. Please wait.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setAnalysisResult(null);

    const systemPrompt = "You are a specialized AI designed for preliminary analysis of retina scans. Analyze the provided fundus image. Look specifically for signs suggestive of common retinal pathologies like Diabetic Retinopathy (microaneurysms, hemorrhages, exudates), Glaucoma (optic disc cupping), and Macular Degeneration. Provide a clear, professional, and concise summary of your key findings, followed by a general recommendation. Use Markdown formatting for readability. Emphasize that this is an AI model analysis and is NOT a substitute for a professional medical diagnosis.";
    
    // The user query that directs the analysis
    const userQuery = "Please perform a detailed, specialized analysis of this retina scan image. What are the most significant observations?";

    const payload = {
      contents: [{
        parts: [
          { text: userQuery },
          {
            inlineData: {
              mimeType: mimeType,
              data: base64Image
            }
          }
        ]
      }],
      systemInstruction: {
        parts: [{ text: systemPrompt }]
      }
    };

    const apiKey = ""; // Canvas environment provides API key automatically when empty
    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${apiKey}`;

    // Helper for exponential backoff retry logic
    const fetchWithRetry = async (url, options, maxRetries = 3) => {
      for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
          const response = await fetch(url, options);
          if (response.ok) {
            return response;
          }
          if (response.status === 429 && attempt < maxRetries - 1) { // Rate limit
            const delay = Math.pow(2, attempt) * 1000 + Math.random() * 1000;
            await new Promise(resolve => setTimeout(resolve, delay));
            continue;
          }
          throw new Error(`API call failed with status: ${response.status}`);
        } catch (e) {
          if (attempt === maxRetries - 1) throw e;
        }
      }
    };

    try {
      const response = await fetchWithRetry(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const result = await response.json();
      const text = result.candidates?.[0]?.content?.parts?.[0]?.text;

      if (text) {
        setAnalysisResult(text);
      } else {
        setError("Analysis failed. Could not parse response from AI model.");
        console.error("Full API Response:", result);
      }
    } catch (e) {
      setError(`An error occurred during API communication: ${e.message}`);
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  }, [isAuthReady]);

  // Handle file selection and start the analysis
  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setImageFile(file);

    try {
      const base64 = await fileToBase64(file);
      await analyzeRetinaScan(base64, file.type);
    } catch (e) {
      setError("Failed to process image file.");
      setIsLoading(false);
      console.error(e);
    }
  };

  // Drag and drop handlers
  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileChange({ target: { files: [file] } });
    } else {
        setError("Invalid file type. Please drop an image file.");
    }
    e.currentTarget.classList.remove('border-indigo-500', 'bg-indigo-50');
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('border-indigo-500', 'bg-indigo-50');
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('border-indigo-500', 'bg-indigo-50');
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4 sm:p-8 font-sans">
      <div className="max-w-4xl mx-auto">
        
        {/* Header */}
        <header className="text-center mb-10">
          <h1 className="text-4xl font-extrabold text-gray-900 tracking-tight">
            EyeHealth AI Retina Scan Analyzer
          </h1>
          <p className="mt-2 text-lg text-gray-600">
            Upload a fundus image for specialized, preliminary analysis using the Gemini Vision model.
          </p>
          {userId && (
              <p className="mt-1 text-sm text-gray-500">
                  User ID: <code className="bg-gray-200 px-2 py-0.5 rounded text-xs">{userId}</code>
              </p>
          )}
        </header>

        <main className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Image Uploader Card */}
          <div className="bg-white p-6 rounded-xl shadow-lg h-full flex flex-col">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">1. Upload Retina Scan</h2>
            
            {/* Dropzone Area */}
            <div
              className={`border-4 border-dashed rounded-xl p-8 transition duration-200 ${
                imageFile ? 'border-green-400 bg-green-50' : 'border-gray-300 hover:border-indigo-400 hover:bg-indigo-50'
              }`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
            >
              <label htmlFor="file-upload" className="block cursor-pointer">
                <div className="text-center">
                  <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 16m-2-2a2 2 0 100-4 2 2 0 000 4z" />
                  </svg>
                  <p className="mt-1 text-sm text-gray-600">
                    <span className="font-medium text-indigo-600 hover:text-indigo-500">
                      Click to upload
                    </span> or drag and drop
                  </p>
                  <p className="text-xs text-gray-500">
                    PNG, JPG, or JPEG file, max 10MB (Retina/Fundus Image)
                  </p>
                </div>
                <input
                  id="file-upload"
                  type="file"
                  accept="image/png, image/jpeg"
                  className="sr-only"
                  onChange={handleFileChange}
                  disabled={isLoading}
                />
              </label>
            </div>

            {/* Selected File Preview */}
            {imageFile && (
              <div className="mt-4 p-4 border border-gray-200 rounded-lg bg-gray-50">
                <h3 className="text-lg font-medium text-gray-800">Selected Image:</h3>
                <p className="text-sm text-gray-600 truncate">{imageFile.name}</p>
                {/* Optional: Add a simple preview image if needed */}
                <div className="mt-3 max-h-40 overflow-hidden rounded-md">
                    <img
                        src={URL.createObjectURL(imageFile)}
                        alt="Retina Scan Preview"
                        className="w-full object-cover rounded-md"
                    />
                </div>
              </div>
            )}
          </div>
          
          {/* Analysis Result Card */}
          <div className="bg-white p-6 rounded-xl shadow-lg flex flex-col">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">2. AI Analysis Results</h2>
            
            {/* Status & Error */}
            {isLoading && (
              <div className="flex items-center space-x-2 text-indigo-600 p-4 bg-indigo-50 rounded-lg">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span className="font-medium">Analyzing scan... Please wait.</span>
              </div>
            )}
            
            {error && (
              <div className="p-4 bg-red-100 border-l-4 border-red-500 text-red-700 rounded-lg" role="alert">
                <p className="font-bold">Analysis Error</p>
                <p className="text-sm">{error}</p>
              </div>
            )}

            {/* Result Display */}
            <div className="mt-4 flex-grow overflow-y-auto">
              {analysisResult ? (
                <div className="prose max-w-none p-4 border border-gray-200 rounded-lg bg-gray-50">
                  <div dangerouslySetInnerHTML={{ __html: analysisResult }} />
                </div>
              ) : (
                <div className="p-6 text-center text-gray-500 border-2 border-dashed border-gray-200 rounded-lg h-full flex items-center justify-center">
                  {imageFile ? (
                    <p>Image ready for analysis. Uploading the image automatically triggers the process.</p>
                  ) : (
                    <p>Upload a retina scan image to begin the specialized AI analysis.</p>
                  )}
                </div>
              )}
            </div>
            
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;