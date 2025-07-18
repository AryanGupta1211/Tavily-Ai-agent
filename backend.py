# Step1 : Setup Pydantic Model (Schema Validation)
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
from ai_agent import get_response_from_ai_agent

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

ALLOWED_MODEL_LIST = ["llama3-70b-8192", "llama-3.3-70b-versatile", "gpt-4o-mini"]

#Step2 : Setup Ai agent from frontend request
app = FastAPI(title="LangGraph AI Agent")

@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API endpoint to interact with the ChatBot using LangGraph and search tools.
    It dynamically selects the model specified in the request.
    """
    
    if request.model_name not in ALLOWED_MODEL_LIST:
        return {"error": "Kindly select valid model id"}
    
    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider
    
    response = get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider)
    
    return response


#Step3 : Run app & Explore Swagger UI Docs
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)