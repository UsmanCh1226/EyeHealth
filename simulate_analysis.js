import fs from 'fs';
import path from 'path';
import fetch from 'node-fetch';

// --- CONFIGURATION ---
// !! REPLACE WITH YOUR ACTUAL API KEY !!
const API_KEY = "AIzaSyCMoWnnwzcWXH60EQdoAEFfUal1Tk-U3as"; 
const MODEL_NAME = "gemini-2.5-flash-preview-09-2025";
const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL_NAME}:generateContent?key=${API_KEY}`;
const IMAGE_FILE_PATH = 'retina_scan.jpeg'; // **UPDATED to .jpeg**
const IMAGE_MIME_TYPE = 'image/jpeg'; // This MIME type is correct for JPEG/JPG

// --- System Prompt (Matches the one in RetinaAnalyzer.jsx) ---
const systemPrompt = `You are a specialized AI assistant for ophthalmology. 
Your task is to analyze the provided retinal fundus image and provide a concise, professional, and clear assessment. 
Focus your analysis on identifying signs of common eye diseases such as Diabetic Retinopathy (DR), Glaucoma, Age-related Macular Degeneration (AMD), or hypertensive changes.
Based on the visual evidence, provide a 'Classification' and a 'Confidence Score'.

Output Structure (Mandatory):
1. Classification: A concise diagnostic statement (e.g., 'Mild Non-proliferative Diabetic Retinopathy', 'Apparent Glaucomatous Optic Neuropathy', 'Healthy Retina').
2. Description: A brief, single-paragraph explanation of the key visual findings (e.g., microaneurysms, hemorrhages, cup-to-disc ratio).
3. Recommendation: A recommendation for the next clinical step (e.g., 'Refer to Ophthalmologist', 'Monitor Annually', 'Immediate Follow-up Required').
`;


// --- Helper function to convert local file to Base64 ---
function fileToBase64(filePath) {
    if (!fs.existsSync(filePath)) {
        throw new Error(`File not found at path: ${filePath}`);
    }
    const fileBuffer = fs.readFileSync(filePath);
    return fileBuffer.toString('base64');
}

// --- Main Analysis Function ---
async function runAnalysis() {
    console.log(`\nStarting analysis for image: ${IMAGE_FILE_PATH}`);

    if (API_KEY === "YOUR_GEMINI_API_KEY_HERE") {
        console.error("FATAL ERROR: Please replace 'YOUR_GEMINI_API_KEY_HERE' in the script with your actual API key.");
        return;
    }

    let base64Image;
    try {
        base64Image = fileToBase64(path.resolve(IMAGE_FILE_PATH));
    } catch (error) {
        console.error(`\nError preparing image: ${error.message}`);
        console.log("Please ensure the image path is correct and the file exists.");
        return;
    }

    const payload = {
        contents: [
            {
                role: "user",
                parts: [
                    { text: "Analyze this retinal fundus image for disease classification." },
                    {
                        inlineData: {
                            mimeType: IMAGE_MIME_TYPE,
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

    console.log("Sending request to Gemini API...");
    try {
        // Implement simple exponential backoff for retries
        const maxRetries = 3;
        for (let attempt = 0; attempt < maxRetries; attempt++) {
            let response;
            try {
                response = await fetch(API_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
            } catch (networkError) {
                // If network error, wait and retry
                if (attempt < maxRetries - 1) {
                    const delay = Math.pow(2, attempt) * 1000;
                    // console.warn(`Network error, retrying in ${delay / 1000}s...`); // Don't log retries
                    await new Promise(resolve => setTimeout(resolve, delay));
                    continue; // Skip to next attempt
                }
                throw networkError; // Throw on final failure
            }
            
            // If response is received, process it
            const result = await response.json();

            if (!response.ok) {
                // Handle API errors (e.g., 400 Bad Request)
                console.error("\n--- API ERROR RESPONSE ---");
                console.error(`HTTP Status: ${response.status}`);
                console.error("Details:", JSON.stringify(result, null, 2));
                return;
            }

            const text = result.candidates?.[0]?.content?.parts?.[0]?.text;

            if (text) {
                console.log("\n--- AI ANALYSIS SUCCESS ---");
                console.log(text);
                return; // Exit successfully
            } else {
                console.log("\n--- WARNING: No text output received ---");
                console.log("Full JSON response:", JSON.stringify(result, null, 2));
                return; // Exit after printing warning/full response
            }
        }


    } catch (e) {
        console.error("\n--- NETWORK/FETCH ERROR ---");
        console.error(e.message);
    }
}

runAnalysis();