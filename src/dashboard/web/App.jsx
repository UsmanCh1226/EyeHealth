import React, { useState, useEffect, useCallback } from 'react';

// --- Firebase Imports ---
import { initializeApp } from 'firebase/app';
import { getAuth, signInWithCustomToken, signInAnonymously } from 'firebase/auth';
import { getFirestore, setLogLevel, doc, setDoc, collection, query, onSnapshot } from 'firebase/firestore';

// Define the Gemini model name and API key placeholder
const GEMINI_MODEL = 'gemini-2.5-flash-preview-09-2025';
const apiKey = ""; // Canvas will provide the key if this is empty

// Function to generate a simple, extremely robust UUID fallback 
const generateFallbackUUID = () => {
    let dt = new Date().getTime();
    const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = (dt + Math.random() * 16) % 16 | 0;
        dt = Math.floor(dt / 16);
        return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
    return uuid;
};

// Inline SVG Spinner Component
const LoadingSpinner = () => (
    <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
);


// Helper function to convert a File object to a Base64 string
const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            // Only need the base64 data part after the comma
            const base64Data = reader.result.split(',')[1];
            if (base64Data) {
                resolve(base64Data);
            } else {
                reject(new Error("Failed to extract Base64 data from file."));
            }
        }; 
        reader.onerror = error => reject(error);
        reader.readAsDataURL(file);
    });
};

// --- Main App Component ---
const App = () => {
    // --- State Management ---
    const [db, setDb] = useState(null);
    const [auth, setAuth] = useState(null);
    const [userId, setUserId] = useState(null);
    const [isAuthReady, setIsAuthReady] = useState(false);
    
    const [imageFile, setImageFile] = useState(null);
    const [imagePreview, setImagePreview] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [history, setHistory] = useState([]);


    // --- Firestore Initialization and Authentication ---
    useEffect(() => {
        const initFirebase = async () => {
            try {
                // Defensive access to global variables
                const configString = typeof __firebase_config !== 'undefined' ? __firebase_config : '{}';
                const firebaseConfig = JSON.parse(configString);
                const customToken = typeof __initial_auth_token !== 'undefined' ? __initial_auth_token : null;

                // Initialize the app
                const app = initializeApp(firebaseConfig);
                const firestore = getFirestore(app);
                const firebaseAuth = getAuth(app);
                setDb(firestore);
                setAuth(firebaseAuth);
                setLogLevel('Debug'); 

                // Handle authentication sign-in
                if (customToken) {
                    await signInWithCustomToken(firebaseAuth, customToken);
                } else {
                    await signInAnonymously(firebaseAuth);
                }

                // Set up auth state change listener to capture the user ID
                const unsubscribe = firebaseAuth.onAuthStateChanged(user => {
                    if (user) {
                        setUserId(user.uid);
                    } else {
                        // Use the robust fallback UUID generator if user is not authenticated
                        setUserId(generateFallbackUUID());
                    }
                    setIsAuthReady(true);
                });

                return () => unsubscribe();
            } catch (err) {
                // Log and update error state if Firebase fails to initialize
                console.error("Firebase initialization or authentication failed:", err);
                setError(`Failed to connect to backend services: ${err.message}.`);
                setIsAuthReady(true); // Must set true to prevent infinite loading state
            }
        };

        // Start initialization
        initFirebase();
        // The dependency array is empty because this should only run once on mount
    }, []);

    // --- Firestore Real-time Listener for Analysis History ---
    useEffect(() => {
        // Guard clause: prevent query from running before auth is ready and services are available
        if (!isAuthReady || !db || !userId) return;

        const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';
        // Private collection path
        const collectionPath = `artifacts/${appId}/users/${userId}/retina_analysis`;
        const q = query(collection(db, collectionPath));

        // Listen for real-time updates
        const unsubscribe = onSnapshot(q, (snapshot) => {
            const fetchedHistory = [];
            snapshot.forEach((doc) => {
                fetchedHistory.push({ ...doc.data(), id: doc.id });
            });
            // Sort history by timestamp (newest first)
            setHistory(fetchedHistory.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0)));
        }, (error) => {
            console.error("Error fetching history: ", error);
            // This error does not crash the app, but informs the user
        });

        // Cleanup listener on unmount
        return () => unsubscribe();
    }, [isAuthReady, db, userId]);

    // --- API Call Logic (Memoized) ---
    const analyzeRetina = useCallback(async (base64Data, mimeType) => {
        if (!db || !userId) {
            setError("Authentication not ready. Please wait.");
            return;
        }

        setIsLoading(true);
        setError(null);
        setAnalysisResult(null);

        // System prompt to guide the model's response
        const systemPrompt = 
            "You are a specialized AI assistant for preliminary retina image analysis. " +
            "Analyze the provided fundus (retina) image for common conditions. " +
            "Provide your analysis in a clear, structured JSON format based on the provided schema. " +
            "DO NOT use any external knowledge search (Google Search). Only use the image provided. " +
            "If the image is not a retina scan, set 'isRetinaImage' to false.";
        
        // User query focused on the task
        const userQuery = 
            "Analyze the retina image. Identify the presence and severity (Mild, Moderate, Severe, or N/A) " +
            "of the following conditions: Diabetic Retinopathy, Glaucoma, Macular Degeneration. " +
            "If none are visible, state 'Healthy Appearance'. Provide a single-paragraph summary of findings in the 'summary' field.";

        // JSON Schema for structured output
        const responseSchema = {
            type: "OBJECT",
            properties: {
                isRetinaImage: { type: "BOOLEAN", description: "True if the uploaded image is a human retina (fundus) scan." },
                summary: { type: "STRING", description: "A concise, single-paragraph summary of the visual findings." },
                findings: {
                    type: "ARRAY",
                    description: "Details of identified conditions.",
                    items: {
                        type: "OBJECT",
                        properties: {
                            condition: { type: "STRING", description: "The specific eye condition (e.g., 'Diabetic Retinopathy')." },
                            severity: { type: "STRING", enum: ["N/A", "Mild", "Moderate", "Severe", "Healthy Appearance"], description: "The estimated severity." },
                            confidence: { type: "STRING", description: "A high-level confidence rating (e.g., 'High', 'Moderate', 'Low')." },
                        },
                        required: ["condition", "severity", "confidence"],
                    }
                }
            },
            required: ["isRetinaImage", "summary", "findings"],
        };
        
        // Function to perform the API call with exponential backoff
        const fetchWithBackoff = async (maxRetries = 5) => {
            const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${apiKey}`;
            const payload = {
                contents: [
                    {
                        role: "user",
                        parts: [
                            { text: userQuery },
                            { inlineData: { mimeType, data: base64Data } }
                        ]
                    }
                ],
                systemInstruction: { parts: [{ text: systemPrompt }] },
                generationConfig: {
                    responseMimeType: "application/json",
                    responseSchema: responseSchema,
                },
            };

            for (let i = 0; i < maxRetries; i++) {
                try {
                    const response = await fetch(apiUrl, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
        
                    if (response.ok) {
                        return response.json();
                    }
                    
                    // Handle rate limiting or server errors by retrying
                    if (response.status === 429 || response.status >= 500) {
                        const delay = Math.pow(2, i) * 1000 + Math.random() * 1000;
                        if (i < maxRetries - 1) {
                            await new Promise(resolve => setTimeout(resolve, delay));
                            continue; // Retry
                        }
                    }
                    throw new Error(`API error: ${response.status} ${response.statusText}`);

                } catch (err) {
                    if (i === maxRetries - 1) {
                        throw err; // Throw final error
                    }
                    // Non-network errors or other transient errors, wait and retry
                    const delay = Math.pow(2, i) * 1000 + Math.random() * 1000;
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        };

        try {
            const result = await fetchWithBackoff();
            const jsonText = result.candidates?.[0]?.content?.parts?.[0]?.text;

            if (!jsonText) {
                throw new Error("Received an empty response from the API.");
            }

            const parsedResult = JSON.parse(jsonText);
            setAnalysisResult(parsedResult);
            
            // --- Save to Firestore ---
            const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';
            const collectionPath = `artifacts/${appId}/users/${userId}/retina_analysis`;
            // Use doc without ID to auto-generate a new ID
            const newDocRef = doc(collection(db, collectionPath)); 
            
            await setDoc(newDocRef, {
                imagePreview: imagePreview, // Save the data URL for history display
                analysis: parsedResult,
                timestamp: Date.now(),
                userId: userId,
                // Do not save the large base64 data to Firestore
            });

        } catch (err) {
            console.error("Gemini API or Firestore Error:", err);
            setError(`Analysis failed: ${err.message}. Please try again.`);
        } finally {
            setIsLoading(false);
        }
    }, [db, userId, imagePreview]);

    // --- Event Handlers ---
    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file && file.type.startsWith('image/')) {
            setImageFile(file);
            // Create a temporary URL for preview
            const reader = new FileReader();
            reader.onload = (e) => setImagePreview(e.target.result);
            reader.readAsDataURL(file);
            setAnalysisResult(null); // Reset previous result
            setError(null);
        } else {
            setImageFile(null);
            setImagePreview(null);
            setError("Please select a valid image file (PNG, JPG).");
        }
    };

    const handleSubmit = async () => {
        if (!imageFile || isLoading || !isAuthReady) return;

        try {
            const base64Data = await fileToBase64(imageFile);
            await analyzeRetina(base64Data, imageFile.type);
        } catch (err) {
            console.error("Image processing error:", err);
            setError("Error processing image: Could not convert file to base64 data.");
            setIsLoading(false);
        }
    };
    
    // Helper to format severity color
    const getSeverityColor = (severity) => {
        switch (severity) {
            case 'Severe': return 'bg-red-100 text-red-800 ring-red-300';
            case 'Moderate': return 'bg-yellow-100 text-yellow-800 ring-yellow-300';
            case 'Mild': return 'bg-blue-100 text-blue-800 ring-blue-300';
            case 'Healthy Appearance': return 'bg-green-100 text-green-800 ring-green-300';
            case 'N/A': return 'bg-gray-100 text-gray-800 ring-gray-300';
            default: return 'bg-gray-100 text-gray-800 ring-gray-300';
        }
    };

    // --- Render Functions ---
    const renderAnalysis = () => {
        if (!analysisResult) return null;

        const { isRetinaImage, summary, findings } = analysisResult;

        if (!isRetinaImage) {
            return (
                <div className="p-6 bg-red-50 border-l-4 border-red-500 text-red-700 rounded-md shadow-inner">
                    <p className="font-bold text-xl mb-2">Analysis Failed: Incorrect Image Type</p>
                    <p>The AI assistant determined the uploaded image is likely <span className="font-semibold underline">NOT</span> a human retina (fundus) scan. Please ensure you upload the correct type of image for analysis.</p>
                </div>
            );
        }
        
        // Trigger an image of a retina fundus diagram for context on the analysis
        // [Image of human eye retina fundus diagram with labels]

        return (
            <div className="space-y-6">
                <h3 className="text-2xl font-bold text-gray-800 border-b pb-2">AI Preliminary Analysis Report</h3>
                
                {/* Summary */}
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg shadow-inner">
                    <p className="font-bold text-blue-700 mb-2">Summary of Findings:</p>
                    <p className="text-gray-700 leading-relaxed italic">{summary}</p>
                </div>
                
                {/* Findings Table */}
                <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200 rounded-xl overflow-hidden shadow-lg">
                        <thead className="bg-indigo-50">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-indigo-700 uppercase tracking-wider">Condition</th>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-indigo-700 uppercase tracking-wider">Severity</th>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-indigo-700 uppercase tracking-wider">Confidence</th>
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-100">
                            {findings.map((f, index) => (
                                <tr key={index} className="hover:bg-gray-50 transition duration-100">
                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{f.condition}</td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <span className={`px-2 inline-flex text-xs leading-5 font-bold rounded-full ring-1 ring-inset ${getSeverityColor(f.severity)}`}>
                                            {f.severity}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{f.confidence}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                <div className="text-xs text-gray-500 p-3 bg-yellow-50 rounded-lg border border-yellow-200">
                    <p className="font-semibold text-yellow-800">DISCLAIMER:</p>
                    <p>This is an AI-generated preliminary analysis and is not a substitute for professional medical advice or diagnosis. Always consult a qualified healthcare provider for eye health concerns.</p>
                </div>
            </div>
        );
    };

    const renderHistory = () => (
        <div className="bg-white p-6 rounded-xl shadow-2xl border border-gray-100 h-full flex flex-col">
            <h2 className="text-2xl font-bold mb-4 text-indigo-800 border-b pb-2">Analysis History</h2>
            {history.length === 0 ? (
                <p className="text-gray-500 italic mt-2">No previous analyses found. Upload an image to start!</p>
            ) : (
                <div className="space-y-4 max-h-[70vh] lg:max-h-[80vh] overflow-y-auto pr-2">
                    {history.map((item) => (
                        <div key={item.id} className="flex items-start p-3 bg-gray-50 rounded-xl border border-gray-200 hover:bg-gray-100 transition duration-150 cursor-pointer"
                             onClick={() => setAnalysisResult(item.analysis)}>
                            <img 
                                src={item.imagePreview} 
                                alt="Retina Scan Preview" 
                                className="w-16 h-16 object-cover rounded-md flex-shrink-0 mr-4 shadow-md border border-gray-300"
                            />
                            <div className="flex-grow min-w-0">
                                <p className="text-xs font-bold text-indigo-700">
                                    {new Date(item.timestamp).toLocaleDateString()} at {new Date(item.timestamp).toLocaleTimeString()}
                                </p>
                                <p className="text-sm text-gray-800 truncate leading-tight mt-1">
                                    {item.analysis?.summary || "Analysis details unavailable."}
                                </p>
                                <div className="mt-1 flex flex-wrap gap-2">
                                    {item.analysis?.findings?.map((f, i) => (
                                        <span key={i} className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${getSeverityColor(f.severity)} ring-1 ring-inset`}>
                                            {f.condition.split(' ')[0]}: {f.severity}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
            <p className="text-sm text-gray-400 mt-6 pt-4 border-t border-gray-100">
                User ID: <span className="font-mono text-xs break-all">{userId || 'Waiting for Auth...'}</span>
            </p>
        </div>
    );
    
    // --- Main UI Render ---
    return (
        <div className="min-h-screen bg-gray-50 font-sans p-4 sm:p-8">
            <header className="text-center mb-8">
                <h1 className="text-4xl sm:text-5xl font-extrabold text-indigo-800">
                    EyeHealth AI Analyzer
                </h1>
                <p className="text-xl text-gray-600 mt-2">
                    Retina Image Analysis Powered by Gemini Vision
                </p>
            </header>

            <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
                
                {/* --- Input and History Column --- */}
                <div className="lg:col-span-1 space-y-8">
                    {renderHistory()}
                </div>

                {/* --- Analysis Column --- */}
                <div className="lg:col-span-2 space-y-8">
                    <div className="bg-white p-6 rounded-xl shadow-2xl border border-indigo-100">
                        <h2 className="text-2xl font-bold mb-4 text-indigo-800 border-b pb-2">Upload Retina Scan</h2>
                        
                        <p className="text-sm text-gray-600 mb-4">
                            Upload a high-quality fundus image (retina scan) to receive a preliminary AI analysis for common conditions like Diabetic Retinopathy, Glaucoma, and Macular Degeneration.
                        </p>
                        
                        {/* File Input */}
                        <div className="mb-6 mt-4">
                            <label className="block text-base font-medium text-gray-700 mb-2">
                                1. Select Fundus Image (.png, .jpg)
                            </label>
                            <input
                                type="file"
                                accept="image/png, image/jpeg, image/jpg"
                                onChange={handleFileChange}
                                className="w-full text-sm text-gray-500
                                    file:mr-4 file:py-2 file:px-4
                                    file:rounded-full file:border-0
                                    file:text-sm file:font-semibold
                                    file:bg-indigo-50 file:text-indigo-700
                                    hover:file:bg-indigo-100 transition duration-150"
                            />
                        </div>

                        {/* Image Preview & Action */}
                        <div className="flex flex-col sm:flex-row gap-6 mb-6 items-center border-t pt-6 border-gray-100">
                            <div className="w-full sm:w-1/2 p-2 border-2 border-dashed border-indigo-300 rounded-xl bg-indigo-50/50 flex justify-center items-center h-48 transition duration-300 hover:border-indigo-400">
                                {imagePreview ? (
                                    <img 
                                        src={imagePreview} 
                                        alt="Uploaded Retina Preview" 
                                        className="max-h-full max-w-full object-contain rounded-lg shadow-xl"
                                    />
                                ) : (
                                    <p className="text-indigo-400 italic font-medium">Image Preview</p>
                                )}
                            </div>

                            <div className="w-full sm:w-1/2 space-y-4">
                                <label className="block text-base font-medium text-gray-700">
                                    2. Run Analysis
                                </label>
                                <button
                                    onClick={handleSubmit}
                                    disabled={!imageFile || isLoading || !isAuthReady}
                                    className={`w-full py-3 px-4 rounded-xl text-white font-semibold transition duration-300 transform flex justify-center items-center ${
                                        !imageFile || isLoading || !isAuthReady
                                            ? 'bg-gray-400 cursor-not-allowed'
                                            : 'bg-indigo-600 hover:bg-indigo-700 shadow-xl shadow-indigo-200 active:scale-[0.98]'
                                    }`}
                                >
                                    {isLoading ? (
                                        <>
                                            <LoadingSpinner />
                                            Analyzing...
                                        </>
                                    ) : 'Start AI Analysis'}
                                </button>
                                {error && (
                                    <p className="mt-3 text-sm text-red-700 bg-red-100 p-3 rounded-lg border border-red-300 font-medium">{error}</p>
                                )}
                                {!isAuthReady && (
                                     <p className="mt-3 text-sm text-yellow-700 bg-yellow-100 p-3 rounded-lg border border-yellow-300 font-medium">
                                        Connecting to Services...
                                    </p>
                                )}
                            </div>
                        </div>

                        {/* Note on Data */}
                         <div className="text-xs text-gray-500 pt-4 border-t border-gray-100">
                             <p>
                                 The system saves the image preview and the analysis report to your private Firestore database history. 
                                 The original high-resolution image data is NOT stored.
                             </p>
                         </div>
                    </div>

                    {/* --- Analysis Output Section --- */}
                    <div className="bg-white p-6 rounded-xl shadow-2xl border border-green-100 min-h-[350px]">
                        {analysisResult ? renderAnalysis() : (
                            <div className="text-center py-16">
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-16 h-16 mx-auto text-indigo-300">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
                                </svg>
                                <p className="mt-4 text-lg text-gray-500">Your AI-powered retina analysis report will appear here after submission.</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default App;