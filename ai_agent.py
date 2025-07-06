# Step1 : Setting up the Groq and Tavily API KEY
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

load_dotenv()


# Load environment variables from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Using only OS module
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


# Step2 : LLM setup and tool setup
groq_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.2)
openai_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

search_tool = TavilySearchResults(max_results=2, api_key=TAVILY_API_KEY)

# Step3 : Search Agent setup
system_prompt = "Act as an AI chatbot who is smart and friendly."

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id) 

    tool = [TavilySearchResults(max_results=2, api_key=TAVILY_API_KEY)] if allow_search else []
    
    agent = create_react_agent(
        model = llm,
        tools = tool,
        state_modifier = system_prompt 
    )
    
    state={"messages": query}
    
    response= agent.invoke(state)
    messages = response.get("messages")
    ai_message = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_message[-1]
