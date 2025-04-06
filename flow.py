import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, TypedDict, Annotated
import operator  # For Annotated in State definition

# Langchain & LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # For checkpointing

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SYSTEM_MESSAGE_CONTENT = (
    "You are a product manager. Ask the user if they have a project. "
    "If they try to ask about something else, redirect them back to product management."
)
print(f"GOOGLE_API_KEY: {GOOGLE_API_KEY}")
print(f"SYSTEM_MESSAGE_CONTENT: {SYSTEM_MESSAGE_CONTENT}")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")

# --- LangGraph Setup ---
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# Node: ask_project – sends initial prompt
def ask_project(state: GraphState):
    print("--- ask_project node ---")
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    current_history = state["messages"]
    messages_to_send = [SystemMessage(content=SYSTEM_MESSAGE_CONTENT)] + current_history
    response = model.invoke(messages_to_send)
    print(f"--- ask_project Response: {response.content[:50]}... ---")
    return {"messages": [response]}

# Branch Node: decision – examines response and calls branch nodes accordingly
def decision(state: GraphState):
    print("--- decision node ---")
    last_response = state["messages"][-1].content.lower()
    if "yes" in last_response:
        print("--- Branching to project_details ---")
        return project_details(state)
    elif "no" in last_response:
        print("--- Branching to redirect_topic ---")
        return redirect_topic(state)
    else:
        print("--- No clear branch. Ending conversation. ---")
        return state

# Branch Node: project_details – for "yes" branch
def project_details(state: GraphState):
    print("--- project_details node ---")
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    prompt = "Great! Please provide details about your project."
    messages_to_send = state["messages"] + [HumanMessage(content=prompt)]
    response = model.invoke(messages_to_send)
    print(f"--- project_details Response: {response.content[:50]}... ---")
    return {"messages": [response]}

# Branch Node: redirect_topic – for "no" branch
def redirect_topic(state: GraphState):
    print("--- redirect_topic node ---")
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    prompt = "It seems you have no project. Let's discuss product management ideas."
    messages_to_send = state["messages"] + [HumanMessage(content=prompt)]
    response = model.invoke(messages_to_send)
    print(f"--- redirect_topic Response: {response.content[:50]}... ---")
    return {"messages": [response]}

# Build the graph: add nodes and set the workflow
workflow = StateGraph(GraphState)
workflow.add_node("ask_project", ask_project)
workflow.add_node("decision", decision)

# Set the entry point and define linear flow
workflow.set_entry_point("ask_project")
workflow.add_edge("ask_project", "decision")
workflow.add_edge("decision", END)

# Compile with MemorySaver for conversation history
memory = MemorySaver()
langgraph_app = workflow.compile(checkpointer=memory)

# --- FastAPI Setup ---
class UserMessage(BaseModel):
    message: str
    conversation_id: str  # For tracking conversation history

class GeminiResponse(BaseModel):
    response: str
    conversation_id: str

app = FastAPI(
    title="LangGraph Gemini API (Branching Flow)",
    description="API with branching flow using LangGraph.",
    version="1.2.0",
)

@app.post("/invoke", response_model=GeminiResponse)
async def invoke_graph(user_input: UserMessage):
    print(f"--- Request (ID: {user_input.conversation_id}): {user_input.message} ---")
    inputs = {"messages": [HumanMessage(content=user_input.message)]}
    config = {"configurable": {"thread_id": user_input.conversation_id}}
    try:
        final_state = langgraph_app.invoke(inputs, config=config)
        ai_response = final_state["messages"][-1]
        if isinstance(ai_response, AIMessage):
            print(f"--- Response Sent (ID: {user_input.conversation_id}) ---")
            return GeminiResponse(
                response=ai_response.content,
                conversation_id=user_input.conversation_id,
            )
        else:
            print(f"--- Error: Last message not from AI (ID: {user_input.conversation_id}) ---")
            raise HTTPException(status_code=500, detail="Graph execution finished unexpectedly.")
    except Exception as e:
        print(f"--- Exception (ID: {user_input.conversation_id}): {e} ---")
        if "GOOGLE_API_KEY" in str(e) or "API key not valid" in str(e):
            raise HTTPException(status_code=500, detail="Server configuration error: Google API Key issue.")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "LangGraph Gemini API (Branching Flow) is running. Use /invoke with JSON: {'message': ..., 'conversation_id': ...}."
    }

if __name__ == "__main__":
    import uvicorn
    print("--- Starting FastAPI server on http://127.0.0.1:8000 ---")
    uvicorn.run(app, host="127.0.0.1", port=8000)
