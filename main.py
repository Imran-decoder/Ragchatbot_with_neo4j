import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain / LangGraph Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.tools import tool, Tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_community import GoogleSearchAPIWrapper
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_core.messages.utils import trim_messages
# 1. Environment Setup
load_dotenv()

# Verify Env Vars (Best Practice for HF Spaces)
required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GOOGLE_API_KEY", "GOOGLE_CSE_ID"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing environment variables: {missing_vars}")

# 2. Database & LLM Setup
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    refresh_schema=True
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True
)

search = GoogleSearchAPIWrapper()


# 3. Define Tools
@tool
def query_graph_database(question: str) -> str:
    """Useful for answering questions about data in the graph database.
    Input should be a specific natural language question."""
    try:
        result = cypher_chain.invoke({"query": question})
        return str(result.get("result", "No result found."))
    except Exception as e:
        return f"Error querying graph: {e}"


google_search_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)

tools = [query_graph_database, google_search_tool]
MAX_MESSAGES = 6

def trim_message_history(state: dict) -> dict:
    """
    LangGraph hook to trim the conversation history to the last MAX_MESSAGES.
    This function runs *before* the LLM is called.
    """
    messages = state.get("messages", [])
    if len(messages) > MAX_MESSAGES:
        # Use trim_messages to cut the history, keeping the latest messages
        trimmed_messages = trim_messages(
            messages,
            strategy="last",
            max_messages=MAX_MESSAGES
        )
        # We return the update, which overwrites the 'messages' in the state
        return {"messages": trimmed_messages}
    return state # Return the original state if no trimming is needed
# 4. Initialize Agent with Memory
# We create a global checkpointer. LangGraph manages isolation via thread_id.
memory = InMemorySaver()

agent_app = create_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    system_prompt=SystemMessage(content="You are a helpful assistant that can query a Neo4j database and search the internet.")
)

# ----------------------------------------------------------------------
# 5. FastAPI App Definition
# ----------------------------------------------------------------------
app = FastAPI(title="Neo4j Gemini Agent")


# -- Data Models --
class ChatRequest(BaseModel):
    query: str
    thread_id: str  # Unique ID for the user session


class ModuleRequest(BaseModel):
    module_name: str
    thread_id: str  # Unique ID for the user session


# -- Helper Function --
def run_agent(message: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    # 1. LOAD FULL HISTORY from checkpoint
    messages: list = []
    try:
        # Get the current state checkpoint for the thread
        current_state = memory.get(config)

        # Extract messages (defaults to an empty list if no history exists)
        messages: list = current_state.get("channel_values", {}).get("messages", [])


    except Exception as e:
        # NOTE: InMemorySaver will typically return the state dictionary even if
        # the key doesn't exist, but using try/except is safer for other checkpointers.
        # Since we initialized messages = [], we can proceed.
        print(f"DEBUG: Could not load memory for {thread_id}. Starting fresh. Error: {e}")

        pass  # messages remains []
        # 2. TRIM HISTORY (keep SystemMessage + last MAX_MESSAGES)
        # The agent's prompt contains the SystemMessage, so we only need to trim the Human/AI messages.
        # The last element will be the new HumanMessage, so we trim based on that.

        # Append the new message
    messages.append(HumanMessage(content=message))

    # Trim, keeping only the last MAX_MESSAGES
    if len(messages) > MAX_MESSAGES:
        # Keep the latest N messages
        trimmed_messages = messages[-MAX_MESSAGES:]
    else:
        trimmed_messages = messages

    # 3. Invoke the agent with the TRIMMED history
    # The LangGraph agent automatically handles saving the new state back to the checkpointer.
    response = agent_app.invoke(
        {"messages": trimmed_messages},
        config=config
    )
    # Return the last message content
    return response['messages'][-1].content


# -- Route 1: Standard Chat --
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Standard chat interface.
    Continuously converses maintaining history based on thread_id.
    """
    try:
        response_text = run_agent(request.query, request.thread_id)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -- Route 2: Module Explanation (Feature 2) --
@app.post("/explain-module")
async def explain_module_endpoint(request: ModuleRequest):
    """
    Triggered when a user enters a specific module.
    1. Generates a concept explanation for the module.
    2. Sets the context in memory so subsequent /chat calls know about it.
    """
    try:
        # Construct the prompt internally
        prompt = f"Please explain the concept of the module: '{request.module_name}' in detail. Start by defining it."

        response_text = run_agent(prompt, request.thread_id)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ResetRequest(BaseModel):
    thread_id: str


@app.delete("/reset-chat")
async def reset_chat_endpoint(request: ResetRequest):
    """Clears the entire conversation history (checkpoint) for a given thread_id."""
    try:
        # InMemorySaver stores data in the 'storage' dictionary
        if request.thread_id in memory.storage:
            del memory.storage[request.thread_id]
            return {"status": "success",
                    "message": f"Conversation history for thread_id '{request.thread_id}' deleted."}
        else:
            return {"status": "success",
                    "message": f"No conversation found for thread_id '{request.thread_id}'. Nothing to delete."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting memory: {e}")


# -- Health Check --
@app.get("/")
def read_root():
    return {"status": "Agent is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)