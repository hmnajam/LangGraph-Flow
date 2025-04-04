import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, TypedDict, Annotated
import operator  # For Annotated in State definition

# Langchain & LangGraph specific imports
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    SystemMessage, # <-- Import SystemMessage
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import (
    MemorySaver, # <-- Import MemorySaver for checkpointing
)

# For loading API key from .env file
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# --- Define Your System Message ---
SYSTEM_MESSAGE_CONTENT = ( # <-- Define your system message here
    "You are a product manager. "
    "Ask the user if they have a project. If they try to ask about something else, "
    "redirect them back to the product management topic."
)
# --- End System Message Definition ---

print(f"GOOGLE_API_KEY: {GOOGLE_API_KEY}")
print(f"SYSTEM_MESSAGE_CONTENT: {SYSTEM_MESSAGE_CONTENT}")


if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")
    # You might want to raise an error or handle this differently in production
    # raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file or environment.")

# --- LangGraph Setup ---

# 1. Define the state for our graph
#    Messages will accumulate in this list (handled by MemorySaver + operator.add)
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


# 2. Define the node that will call the Gemini model
def call_gemini_model(state: GraphState):
    """
    Calls the Gemini LLM with the current state's messages (plus system message)
    and updates the state.
    """
    print("--- Calling Gemini ---")
    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=500, detail="Google API Key not configured on server."
        )

    # Ensure you have the correct model name (e.g., "gemini-1.5-flash", "gemini-pro")
    # Note: "gemini-2.0-flash" might not be a valid public model name yet.
    # Using "gemini-1.5-flash" or "gemini-pro" is safer.
    # Let's use gemini-1.5-flash as an example
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY
    )

    # Get the current conversation history from the state
    # The checkpointer ensures this list contains previous turns
    current_history = state["messages"]
    print(f"--- Current History (State): {current_history} ---")


    # *** Add the System Message ***
    # Prepend the system message to the history before sending to the model
    messages_to_send = [SystemMessage(content=SYSTEM_MESSAGE_CONTENT)] + current_history
    print(f"--- Messages Sent to LLM: {messages_to_send} ---")

    # Invoke the model
    response = model.invoke(messages_to_send)
    print(
        f"--- Gemini Response Received: {response.content[:50]}... ---"
    )  # Log snippet

    # Return the updated state dictionary, adding *only* the AI's response.
    # The `operator.add` in GraphState and the checkpointer handle appending
    # this new message to the persisted state.
    return {"messages": [response]}


# 3. Define the graph structure
workflow = StateGraph(GraphState)

# Add the node
workflow.add_node("gemini_caller", call_gemini_model)

# Set the entry point
workflow.set_entry_point("gemini_caller")

# Add the edge. After calling Gemini, we finish (END).
workflow.add_edge("gemini_caller", END)

# 4. Compile the graph into a runnable application
#    *** Add MemorySaver for checkpointing ***
memory = MemorySaver()
langgraph_app = workflow.compile(checkpointer=memory) # <-- Compile WITH the checkpointer

# --- FastAPI Setup ---

# Define the request body model using Pydantic
class UserMessage(BaseModel):
    message: str
    # *** Add conversation_id (required for history) ***
    conversation_id: str # Use this to track different conversations


# Define the response body model
class GeminiResponse(BaseModel):
    response: str
    conversation_id: str # Also return the ID for clarity


# Create the FastAPI app instance
app = FastAPI(
    title="LangGraph Gemini API (with History)",
    description="An API to send messages to Gemini via LangGraph, maintaining conversation history.",
    version="1.1.0",
)


# Define the API endpoint
@app.post("/invoke", response_model=GeminiResponse)
async def invoke_graph(user_input: UserMessage):
    """
    Receives a user message and conversation_id, processes it through
    the LangGraph/Gemini workflow using checkpointing for history,
    and returns the AI's response.
    """
    print(f"--- Received Request (ID: {user_input.conversation_id}): {user_input.message} ---")

    # Prepare the input for the LangGraph app
    # We only send the *new* user message. The checkpointer loads history.
    inputs = {"messages": [HumanMessage(content=user_input.message)]}

    # *** Configuration for invoking the graph with the specific conversation thread ***
    config = {"configurable": {"thread_id": user_input.conversation_id}}

    try:
        # Invoke the LangGraph application
        # The checkpointer automatically loads history for the thread_id
        # and saves the updated state.
        final_state = langgraph_app.invoke(inputs, config=config)

        # Extract the *last* message, which should be the AI's response
        # added in this invocation step.
        ai_response = final_state["messages"][-1]

        # Ensure it's an AI message and get its content
        if isinstance(ai_response, AIMessage):
            print(f"--- Sending Response (ID: {user_input.conversation_id}) ---")
            return GeminiResponse(
                response=ai_response.content,
                conversation_id=user_input.conversation_id # Return the ID
            )
        else:
            # This indicates an issue, as our simple graph should always end with the AI message
            print(f"--- Error: Last message was not from AI (ID: {user_input.conversation_id}) ---")
            raise HTTPException(
                status_code=500, detail="Graph execution finished unexpectedly."
            )

    except Exception as e:
        # Catch potential errors during graph execution or API key issues
        print(f"--- Error during graph invocation (ID: {user_input.conversation_id}): {e} ---")
        # Check specifically for API key issues if possible, otherwise generic error
        if "GOOGLE_API_KEY" in str(e) or "API key not valid" in str(e):
             # More specific check based on potential Google API error messages
            raise HTTPException(
                status_code=500,
                detail="Server configuration error: Google API Key issue.",
            )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Basic root endpoint for testing if the server is running
@app.get("/")
async def root():
    return {
        "message": "LangGraph Gemini API (with History) is running. Use the /invoke endpoint with a POST request including 'message' and 'conversation_id'."
    }


# --- Running the App (using uvicorn) ---
if __name__ == "__main__":
    import uvicorn

    print("--- Starting FastAPI server on http://127.0.0.1:8000 ---")
    print("--- Send POST requests to http://127.0.0.1:8000/invoke ---")
    print("--- Request body should be JSON like: {\"message\": \"Your message\", \"conversation_id\": \"some-unique-id\"} ---")
    print(
        f"--- Google API Key Loaded: {'Yes' if GOOGLE_API_KEY else 'No! Check .env file'} ---"
    )
    print(f"--- Using System Message: \"{SYSTEM_MESSAGE_CONTENT}\" ---")
    print(f"--- Using Checkpointer: MemorySaver (History persists only while server runs) ---")
    uvicorn.run(app, host="127.0.0.1", port=8000)