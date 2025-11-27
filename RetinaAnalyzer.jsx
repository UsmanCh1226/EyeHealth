import React, { useState, useCallback, useEffect } from 'react';
import { Upload, X, Loader2, Zap, AlertTriangle, CheckCircle, Search, Save, History, BookOpen, Eye } from 'lucide-react';

// --- Firebase Imports ---
import { initializeApp } from 'firebase/app';
import { getAuth, signInAnonymously, signInWithCustomToken, onAuthStateChanged } from 'firebase/auth';
import { 
    getFirestore, 
    doc, 
    addDoc, 
    collection, 
    query, 
    onSnapshot, 
    orderBy, 
    serverTimestamp 
} from 'firebase/firestore';

// --- Global Variables (Provided by Canvas Environment) ---
const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';
const firebaseConfig = typeof __firebase_config !== 'undefined' ? JSON.parse(__firebase_config) : {};
const initialAuthToken = typeof __initial_auth_token !== 'undefined' ? __initial_auth_token : null;

// --- API Configuration ---
const apiKey = ""; 
const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${apiKey}`;

// --- Firestore Collection Paths ---
// Private data stored under the user's ID
const getCollectionPath = (userId) => `/artifacts/${appId}/users/${userId}/retina_reports`;


// --- Helper Component for Results (Moved outside App for clarity) ---
const ResultDisplay = ({ text, onSave, isSaving }) => {
    // Simple regex parsing of the mandatory structured output (Classification, Description, Recommendation)
    // Note: This parsing is robust to handle potential variations in model output.
    const classificationMatch = text.match(/Classification:\s*(.*?)\s*(?:\n|$)/i);
    const descriptionMatch = text.match(/Description:\s*(.*?)\s*(?:\n|$)/i);
    const recommendationMatch = text.match(/Recommendation:\s*(.*?)\s*(?:\n|$)/i);
    
    // Extract values or use placeholders
    const classification = classificationMatch ? classificationMatch[1].trim() : 'N/A';
    const description = descriptionMatch ? descriptionMatch[1].trim() : text; // Fallback to full text
    const recommendation = recommendationMatch ? recommendationMatch[1].trim() : 'Review manually.';

    const isHealthy = classification.toLowerCase().includes('healthy') || classification.toLowerCase().includes('normal');
    const icon = isHealthy ? <CheckCircle className="w-6 h-6 text-green-400 mr-2" /> : <AlertTriangle className="w-6 h-6 text-red-400 mr-2" />;
    const statusColor = isHealthy ? 'text-green-400' : 'text-red-400';
    const statusBg = isHealthy ? 'bg-green-900/50' : 'bg-red-900/50';

    return (
        <div className="space-y-6 mt-8 p-6 bg-gray-700/50 rounded-xl border border-blue-500/50">
            <div className={`p-4 rounded-lg flex items-center justify-between ${statusBg}`}>
                <div className="flex items-center">
                    {icon}
                    <h3 className={`text-xl font-bold ${statusColor}`}>{classification}</h3>
                </div>
                {onSave && (
                    <button
                        onClick={() => onSave({ classification, description, recommendation })}
                        disabled={isSaving}
                        className={`px-4 py-2 rounded-lg font-semibold flex items-center transition duration-200 
                            ${isSaving ? 'bg-gray-500 text-gray-300 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700 text-white'}`}
                    >
                        {isSaving ? (
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        ) : (
                            <Save className="w-4 h-4 mr-2" />
                        )}
                        {isSaving ? 'Saving...' : 'Save Report'}
                    </button>
                )}
            </div>

            <div className="space-y-4">
                <h4 className="text-lg font-semibold text-blue-300 border-b border-blue-500/50 pb-1">
                    Key Visual Findings
                </h4>
                <p className="text-gray-200 leading-relaxed indent-6">{description}</p>
            </div>

            <div className="space-y-4">
                <h4 className="text-lg font-semibold text-blue-300 border-b border-blue-500/50 pb-1">
                    Clinical Recommendation
                </h4>
                <p className="text-yellow-300 font-medium">{recommendation}</p>
            </div>
        </div>
    );
};

// --- History Viewer Component (for saved reports) ---
const HistoryViewer = ({ reports, onViewReport }) => {
    if (reports.length === 0) {
        return <p className="text-gray-500 text-center italic mt-4">No reports saved yet.</p>;
    }
    return (
        <div className="space-y-3 mt-4 max-h-[70vh] overflow-y-auto pr-2">
            {reports.map((report) => (
                <div 
                    key={report.id} 
                    className="p-3 bg-gray-700/50 rounded-lg flex justify-between items-center border border-gray-600 hover:bg-gray-700 cursor-pointer transition"
                    onClick={() => onViewReport(report)}
                >
                    <div>
                        <p className="font-semibold text-white truncate">{report.classification || 'Untitled Report'}</p>
                        <p className="text-xs text-gray-400">
                            {report.timestamp ? new Date(report.timestamp.seconds * 1000).toLocaleString() : 'Saving...'}
                        </p>
                    </div>
                    <BookOpen className="w-5 h-5 text-blue-400" />
                </div>
            ))}
        </div>
    );
};


// --- Component: RetinaAnalyzer (Main App) ---
const App = () => {
    // --- State Management ---
    const [imageFile, setImageFile] = useState(null);
    const [base64Image, setBase64Image] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null); // Raw text result from Gemini
    const [isLoading, setIsLoading] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('analyze'); // 'analyze' or 'history'
    
    // Firebase State
    const [db, setDb] = useState(null);
    const [auth, setAuth] = useState(null);
    const [userId, setUserId] = useState(null);
    const [isAuthReady, setIsAuthReady] = useState(false);
    const [reports, setReports] = useState([]); // Saved reports from Firestore
    const [viewingReport, setViewingReport] = useState(null); // Detailed view of a past report

    // --- System Prompt for Gemini (Defines the AI's role) ---
    const systemPrompt = `You are a specialized AI assistant for ophthalmology. 
        Your task is to analyze the provided retinal fundus image and provide a concise, professional, and clear assessment. 
        Focus your analysis on identifying signs of common eye diseases such as Diabetic Retinopathy (DR), Glaucoma, Age-related Macular Degeneration (AMD), or hypertensive changes.
        Based on the visual evidence, provide a 'Classification' and a 'Confidence Score'.
        
        Output Structure (Mandatory):
        1. Classification: A concise diagnostic statement (e.g., 'Mild Non-proliferative Diabetic Retinopathy', 'Apparent Glaucomatous Optic Neuropathy', 'Healthy Retina').
        2. Description: A brief, single-paragraph explanation of the key visual findings (e.g., microaneurysms, hemorrhages, cup-to-disc ratio).
        3. Recommendation: A recommendation for the next clinical step (e.g., 'Refer to Ophthalmologist', 'Monitor Annually', 'Immediate Follow-up Required').
    `;

    // --- Auth and Firestore Initialization (useEffect with empty dependency array) ---
    useEffect(() => {
        if (!firebaseConfig.apiKey) {
            console.error("Firebase configuration is missing. Cannot initialize Firestore.");
            // We set isAuthReady to true even if config is missing to allow UI to render fully
            // but database operations will fail.
            setIsAuthReady(true);
            return;
        }

        try {
            const app = initializeApp(firebaseConfig);
            const dbInstance = getFirestore(app);
            const authInstance = getAuth(app);
            
            setDb(dbInstance);
            setAuth(authInstance);

            // 1. Authentication
            const unsubscribeAuth = onAuthStateChanged(authInstance, (user) => {
                if (user) {
                    setUserId(user.uid);
                    setIsAuthReady(true);
                } else {
                    // Sign in anonymously if no token is available or user is signed out
                    if (!initialAuthToken) {
                        signInAnonymously(authInstance)
                            .then((anonUserCredential) => {
                                setUserId(anonUserCredential.user.uid);
                            })
                            .catch(error => {
                                console.error("Anonymous sign-in failed:", error);
                                setUserId(crypto.randomUUID()); // Fallback non-auth user ID
                            })
                            .finally(() => setIsAuthReady(true));
                    }
                }
            });

            // Handle custom token sign-in once
            if (initialAuthToken) {
                signInWithCustomToken(authInstance, initialAuthToken)
                    .catch(error => console.error("Custom token sign-in failed:", error));
            }

            return () => {
                unsubscribeAuth();
            };
        } catch (e) {
            console.error("Firebase initialization failed:", e);
        }
    }, []);

    // --- Firestore Real-time Listener for Reports ---
    useEffect(() => {
        // Prevent running if Firebase is not initialized or user is not identified
        if (!isAuthReady || !db || !userId) return;

        console.log(`Setting up listener for user: ${userId}`);
        const reportsRef = collection(db, getCollectionPath(userId));
        
        // Query to get all reports, ordered by creation time
        // NOTE: We rely on client-side sorting if orderBy leads to index errors in the environment.
        const q = query(reportsRef, orderBy("timestamp", "desc"));

        // Set up real-time listener
        const unsubscribe = onSnapshot(q, (snapshot) => {
            const fetchedReports = [];
            snapshot.forEach((doc) => {
                fetchedReports.push({ id: doc.id, ...doc.data() });
            });
            // If the query failed to sort, manually sort by timestamp before saving to state
            fetchedReports.sort((a, b) => (b.timestamp?.seconds || 0) - (a.timestamp?.seconds || 0));
            setReports(fetchedReports);
        }, (error) => {
            console.error("Error listening to reports collection:", error);
            setError("Failed to load report history. Check console for details.");
        });

        // Cleanup the listener on component unmount or dependency change
        return () => unsubscribe();
    }, [db, userId, isAuthReady]); 

    // --- Utility Function: File to Base64 ---
    const fileToBase64 = (file) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result.split(',')[1]); // Resolve only the base64 part
            reader.onerror = error => reject(error);
        });
    };

    // --- Event Handler: File Selection ---
    const handleFileChange = async (event) => {
        const file = event.target.files[0];
        if (file) {
            if (!file.type.startsWith('image/')) {
                setError("Please upload a valid image file.");
                return;
            }
            if (file.size > 5 * 1024 * 1024) { // Max 5MB
                setError("Image file is too large (max 5MB).");
                return;
            }
            setError(null);
            setImageFile(file);
            setAnalysisResult(null); // Clear previous results
            setViewingReport(null);
            setActiveTab('analyze'); // Switch back to analysis tab
            
            try {
                const base64 = await fileToBase64(file);
                setBase64Image(base64);
            } catch (err) {
                setError("Failed to process image file.");
            }
        }
    };

    // --- Core Function: Call Gemini API for Analysis ---
    const analyzeImage = useCallback(async () => {
        if (!base64Image) {
            setError("Please select an image first.");
            return;
        }

        setIsLoading(true);
        setError(null);
        setAnalysisResult(null);

        // Exponential backoff retry logic
        for (let attempt = 0; attempt < 3; attempt++) {
            try {
                const payload = {
                    contents: [
                        {
                            role: "user",
                            parts: [
                                { text: "Analyze this retinal fundus image for disease classification." },
                                {
                                    inlineData: {
                                        mimeType: imageFile.type,
                                        data: base64Image
                                    }
                                }
                            ]
                        }
                    ],
                    systemInstruction: {
                        parts: [{ text: systemPrompt }]
                    },
                };

                const response = await fetch(apiUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    throw new Error(`API call failed with status: ${response.status}`);
                }

                const result = await response.json();
                const text = result.candidates?.[0]?.content?.parts?.[0]?.text;

                if (text) {
                    setAnalysisResult(text);
                    setIsLoading(false);
                    return; // Success, exit retry loop
                } else {
                    throw new Error("Received empty response from the model.");
                }

            } catch (e) {
                console.error(`Attempt ${attempt + 1} failed:`, e);
                if (attempt < 2) {
                    await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000)); // Exponential backoff
                } else {
                    setError(`Analysis failed after multiple retries: ${e.message}`);
                    setIsLoading(false);
                }
            }
        }
    }, [base64Image, imageFile?.type, systemPrompt]);


    // --- Firestore Save Handler ---
    const saveReport = async (parsedData) => {
        if (!db || !userId) {
            setError("Database not ready. Please wait for initialization.");
            return;
        }

        setIsSaving(true);
        setError(null);

        // NOTE: We use a placeholder URI as we cannot store large base64 strings in Firestore.
        // In a real application, the image would be uploaded to Firebase Storage.
        const imagePlaceholderUri = `data:${imageFile.type};base64,...(redacted for storage)`; 

        const reportData = {
            ...parsedData,
            imagePreviewUri: imagePlaceholderUri,
            originalImageFileName: imageFile.name,
            timestamp: serverTimestamp(),
            userId: userId,
        };

        try {
            const reportsCollection = collection(db, getCollectionPath(userId));
            await addDoc(reportsCollection, reportData);
            console.log("Report saved successfully!");
            // Clear the analysis result and image data after successful saving
            setAnalysisResult(null);
            setBase64Image(null);
            setImageFile(null);
        } catch (e) {
            console.error("Error saving document:", e);
            setError("Failed to save report to database.");
        } finally {
            setIsSaving(false);
        }
    };

    // --- Report Display Logic ---
    const displayReportContent = () => {
        if (activeTab === 'history') {
            if (viewingReport) {
                // Display specific historical report
                // Reconstruct the text for the ResultDisplay component
                const tempText = `Classification: ${viewingReport.classification}\nDescription: ${viewingReport.description}\nRecommendation: ${viewingReport.recommendation}`;
                return (
                    <>
                        <button 
                            onClick={() => setViewingReport(null)}
                            className="text-blue-400 hover:text-blue-300 flex items-center mb-4 transition text-sm"
                        >
                            <span className="text-xl font-bold">← Back to History</span>
                        </button>
                        <div className="text-center mb-4 p-4 bg-gray-700/50 rounded-lg">
                            <h3 className="text-lg text-white">Report from: {new Date(viewingReport.timestamp.seconds * 1000).toLocaleString()}</h3>
                            <p className="text-sm text-gray-400">File Analyzed: {viewingReport.originalImageFileName}</p>
                        </div>
                        <div className="relative p-2 bg-gray-800 rounded-xl border border-gray-700 shadow-lg mb-6">
                            <div className="aspect-video bg-black rounded-lg overflow-hidden flex flex-col items-center justify-center h-48">
                                <Eye className="w-16 h-16 text-gray-700 mb-2" />
                                <p className="text-gray-400">Image Preview Not Stored</p>
                                <p className="text-xs text-gray-500">The full image was not saved to the database due to size limits.</p>
                            </div>
                        </div>
                        {/* We pass null for onSave and false for isSaving since historical reports cannot be re-saved */}
                        <ResultDisplay text={tempText} onSave={null} isSaving={false} />
                    </>
                );
            }
            // Display history list
            return <HistoryViewer reports={reports} onViewReport={setViewingReport} />;

        } else { // activeTab === 'analyze'
            if (!analysisResult && !isLoading) {
                return (
                    <p className="text-gray-500 text-center italic">
                        Upload an image and click "Run AI Analysis" to generate the report.
                    </p>
                );
            } else if (isLoading) {
                return (
                    <div className="text-center text-blue-400">
                        <Loader2 className="w-10 h-10 mx-auto animate-spin mb-3" />
                        <p className="font-semibold">Processing image via Gemini Vision Model...</p>
                    </div>
                );
            } else if (analysisResult) {
                // Display current analysis result
                return <ResultDisplay text={analysisResult} onSave={saveReport} isSaving={isSaving} />;
            }
        }
        return null;
    };


    return (
        <div className="min-h-screen bg-gray-900 text-white p-4 sm:p-8 font-sans">
            <script src="https://cdn.tailwindcss.com"></script>
            <div className="max-w-4xl mx-auto">
                
                <header className="text-center mb-8 p-6 bg-gray-800 rounded-xl shadow-xl border border-blue-600/50">
                    <Zap className="w-10 h-10 mx-auto text-cyan-400 mb-2" />
                    <h1 className="text-3xl font-extrabold text-cyan-300">
                        Ocular AI Diagnostic Assistant
                    </h1>
                    <p className="text-gray-400 mt-1 flex justify-center items-center">
                        Simulated Machine Learning Inference | User ID: 
                        <span className="ml-2 px-2 py-0.5 bg-gray-700 text-xs rounded font-mono text-yellow-300">
                            {userId || 'Authenticating...'}
                        </span>
                    </p>
                </header>

                {/* Tab Navigation */}
                <div className="flex mb-6 border-b border-gray-700">
                    <button
                        onClick={() => { setActiveTab('analyze'); setViewingReport(null);}}
                        className={`px-4 py-2 flex items-center font-medium transition duration-200 ${
                            activeTab === 'analyze' 
                                ? 'text-blue-400 border-b-2 border-blue-400' 
                                : 'text-gray-400 hover:text-white'
                        }`}
                    >
                        <Search className="w-4 h-4 mr-2" /> New Analysis
                    </button>
                    <button
                        onClick={() => { setActiveTab('history'); setAnalysisResult(null);}}
                        className={`px-4 py-2 flex items-center font-medium transition duration-200 ${
                            activeTab === 'history' 
                                ? 'text-blue-400 border-b-2 border-blue-400' 
                                : 'text-gray-400 hover:text-white'
                        }`}
                    >
                        <History className="w-4 h-4 mr-2" /> Report History ({reports.length})
                    </button>
                </div>


                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    
                    {/* --- LEFT: Image Upload and Preview (Only visible in Analyze tab) --- */}
                    {activeTab === 'analyze' && (
                        <div className="space-y-6">
                            <h2 className="text-xl font-bold text-blue-400">1. Upload Retinal Image</h2>
                            
                            <div className="p-6 bg-gray-800 rounded-xl border border-gray-700 shadow-lg">
                                <label 
                                    htmlFor="file-upload" 
                                    className={`flex flex-col items-center justify-center w-full h-48 border-2 border-dashed rounded-lg cursor-pointer transition duration-300 
                                        ${imageFile ? 'border-green-500 hover:bg-gray-700' : 'border-blue-500 hover:border-blue-300 hover:bg-gray-700'}`
                                    }
                                >
                                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                                        {imageFile ? (
                                            <>
                                                <CheckCircle className="w-8 h-8 text-green-500 mb-2" />
                                                <p className="mb-2 text-sm text-green-300 font-medium">Image Loaded Successfully</p>
                                                <p className="text-xs text-gray-400">{imageFile.name}</p>
                                            </>
                                        ) : (
                                            <>
                                                <Upload className="w-8 h-8 mb-2 text-blue-400" />
                                                <p className="mb-2 text-sm text-gray-300"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                                                <p className="text-xs text-gray-500">Fundus Retinal Scan (Max 5MB)</p>
                                            </>
                                        )}
                                    </div>
                                    <input id="file-upload" type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
                                </label>
                            </div>

                            {base64Image && (
                                <div className="relative p-2 bg-gray-800 rounded-xl border border-gray-700 shadow-lg">
                                    <h3 className="text-center text-gray-400 mb-2">Image Preview</h3>
                                    <div className="aspect-video bg-black rounded-lg overflow-hidden flex items-center justify-center">
                                        <img 
                                            src={`data:${imageFile.type};base64,${base64Image}`} 
                                            alt="Retinal Scan Preview" 
                                            className="w-full h-auto object-contain max-h-96 rounded-lg shadow-md" 
                                        />
                                    </div>
                                    <button
                                        onClick={() => {setImageFile(null); setBase64Image(null); setAnalysisResult(null); setViewingReport(null);}}
                                        className="absolute top-4 right-4 p-1 bg-red-600 rounded-full hover:bg-red-700 transition"
                                        title="Remove Image"
                                    >
                                        <X className="w-5 h-5 text-white" />
                                    </button>
                                </div>
                            )}
                            
                            {error && (
                                <div className="p-3 bg-red-900/50 border border-red-500 rounded-lg text-red-300 font-medium flex items-center">
                                    <AlertTriangle className="w-5 h-5 mr-2" />
                                    {error}
                                </div>
                            )}

                            <button
                                onClick={analyzeImage}
                                disabled={!base64Image || isLoading || !isAuthReady || isSaving}
                                className={`w-full py-3 rounded-xl font-bold text-lg flex items-center justify-center transition duration-300 
                                    ${!base64Image || isLoading || !isAuthReady || isSaving
                                        ? 'bg-gray-600 text-gray-400 cursor-not-allowed' 
                                        : 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-500/50'}`
                                }
                            >
                                {isLoading ? (
                                    <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Analyzing...</>
                                ) : !isAuthReady ? (
                                    <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Connecting...</>
                                ) : (
                                    <><Search className="w-5 h-5 mr-2" /> Run AI Analysis</>
                                )}
                            </button>
                        </div>
                    )}
                    

                    {/* --- RIGHT: Analysis Results or History --- */}
                    <div className={`space-y-6 ${activeTab === 'history' ? 'lg:col-span-2' : ''}`}>
                        <h2 className="text-xl font-bold text-blue-400">
                            {activeTab === 'analyze' ? '2. AI Diagnostic Report' : '3. Saved Report History'}
                        </h2>
                        
                        <div className={`p-6 bg-gray-800 rounded-xl border border-gray-700 shadow-lg flex flex-col justify-start ${activeTab === 'history' ? 'min-h-0' : 'min-h-96 items-center'}`}>
                            {displayReportContent()}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default App;