import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List
import pathlib

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not found in .env file")
    print("Please add your API key to .env file")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured")

app = FastAPI(title="Free Chatbot")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []

class ChatResponse(BaseModel):
    response: str
    error: str | None = None

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        if not GEMINI_API_KEY:
            return ChatResponse(response="", error="API key not configured. Please add GEMINI_API_KEY to .env file")
        
        model = genai.GenerativeModel('gemini-pro')
        
        # Prepare history
        chat_history = []
        for msg in request.history:
            chat_history.append({
                "role": msg.role,
                "parts": [msg.content]
            })
        
        chat_session = model.start_chat(history=chat_history)
        response = chat_session.send_message(request.message)
        
        return ChatResponse(response=response.text)
    
    except Exception as e:
        return ChatResponse(response="", error=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

# Serve frontend
frontend_path = pathlib.Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Chatbot server running at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)