---
title: Chatmodel
emoji: 📉
colorFrom: indigo
colorTo: indigo
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


---
## 📌 **1. What This Application Is**

This file implements a **FastAPI-based chatbot backend** that:

1. Uses **Neo4j as a knowledge graph**
2. Uses a **large language model** (Gemini) via **LangChain / LangGraph**
3. Includes **a conversational agent that can query both Neo4j and Google Search**
4. Keeps **conversation history per session (thread ID)**

This makes it a **retrieval-augmented chatbot** capable of combining structured graph data with external search and conversational memory. ([GitHub][1])

---

## 🧱 **2. Environment Setup**

```python
load_dotenv()
required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GOOGLE_API_KEY", "GOOGLE_CSE_ID"]
```

Before the chatbot runs, it ensures environment variables for:

✔ Neo4j credentials
✔ Google API keys (for search)

If any are missing, the app **raises an error**. ([GitHub][1])

---

## 🔗 **3. Graph & Model Configuration**

### Neo4j Graph

```python
graph = Neo4jGraph(
  url=os.getenv("NEO4J_URI"),
  username=os.getenv("NEO4J_USERNAME"),
  password=os.getenv("NEO4J_PASSWORD"),
  refresh_schema=True
)
```

* Connects to a **Neo4j database**
* `refresh_schema=True` updates the graph schema (nodes/relationships) in memory

### LLM Setup

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
cypher_chain = GraphCypherQAChain.from_llm(llm=llm, graph=graph, ...)
```

* Loads **Google’s Gemini model** from LangChain
* Prepares a **Cypher query answering chain** that translates language questions into Cypher queries for the graph backend ([GitHub][1])

### Google Search

```python
search = GoogleSearchAPIWrapper()
```

This is a **tool to fetch search results** from Google (used as a fallback or context tool). ([GitHub][1])

---

## 🔧 **4. Tools for the Agent**

### Graph Query Tool

```python
@tool
def query_graph_database(question: str) -> str:
    result = cypher_chain.invoke({"query": question})
```

This lets the agent ask **natural language questions and get graph responses**. ([GitHub][1])

### Google Search Tool

```python
google_search_tool = Tool(
    name="google_search",
    description="Search Google",
    func=search.run,
)
```

Enables the agent to augment its knowledge using Google Search results. ([GitHub][1])

### Tool List

```python
tools = [query_graph_database, google_search_tool]
```

Both are exposed to the agent as capabilities. ([GitHub][1])

---

## 🧠 **5. Conversation Memory & Trimming**

There’s an in-memory session history for each chat thread:

```python
memory = InMemorySaver()
```

And a helper to trim histories when they get too long:

```python
def trim_message_history(state: dict) -> dict:
    ...
```

This keeps interactions short and relevant. ([GitHub][1])

---

## 📦 **6. Agent Initialization**

```python
agent_app = create_agent( model=llm, tools=tools, checkpointer=memory, system_prompt=... )
```

* Creates the actual **chat agent**
* Annualizes conversation with tools + memory
* Pinned System Prompt tells it:

  > *“You are a helpful assistant that can query a Neo4j database and search the internet.”* ([GitHub][1])

---

## 🚀 **7. FastAPI Application & API Routes**

👇 Below are **each route**, what it does, and how to use it.

---

### 🗣️ **POST /chat**

**Purpose:** Core chat endpoint
**Input JSON:**

```json
{
  "query": "Your question text",
  "thread_id": "unique_session_id"
}
```

**What it does:**

* Uses the agent to respond to `query`
* Maintains history per `thread_id`
* Chatbot can use:

  * Neo4j graph queries
  * Google Search
  * Conversation memory

**Response:**

```json
{"response": "Agent answer text"}
```

**Example:**

````
POST /chat
{
  "query": "What is the capital of France?",
  "thread_id": "session123"
}
``` :contentReference[oaicite:10]{index=10}

---

### 📘 **POST /explain-module**

**Purpose:** Explain a concept/module by name  
**Input JSON:**

```json
{
  "module_name": "Graph Theory",
  "thread_id": "session123"
}
````

**What it does:**

* Internally constructs a prompt:

  > “Please explain the concept of the module: 'X’…”
* Sends that to the agent
* Useful for teaching/explaining specific modules in context

**Response:**

````json
{"response": "Detailed explanation text"}
``` :contentReference[oaicite:11]{index=11}

---

### ♻️ **DELETE /reset-chat**

**Purpose:** Clears conversation history for a thread  
**Input JSON:**

```json
{"thread_id": "session123"}
````

**Action:**

* Removes memory for that thread if exists
* Resets further chats to start fresh

**Response:**

````json
{"status": "success", "message": "Conversation history deleted."}
``` :contentReference[oaicite:12]{index=12}

---

### ✅ **GET /**

**Purpose:** Simple health check

**Response:**

```json
{"status": "Agent is running"}
``` :contentReference[oaicite:13]{index=13}

---

## 🏁 **8. Main Startup**

```python
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
````

Starts the FastAPI server on **port 7860** by default. ([GitHub][1])

---

## 🤖 **9. How the Agent Works Internally**

When you call `/chat`:

1. The saved history is loaded (based on `thread_id`)
2. Messages are trimmed if too long
3. The new user message is appended
4. The agent processes this with memory + tools
5. The agent returns the last LLM message

Thus the bot can reference prior conversation and tools in its answer. ([GitHub][1])

---

## 📌 **Summary of API Endpoints**

| Method     | Endpoint          | Purpose                     |               |
| ---------- | ----------------- | --------------------------- | ------------- |
| **GET**    | `/`               | Health check                |               |
| **POST**   | `/chat`           | Chatbot conversation        |               |
| **POST**   | `/explain-module` | Get conceptual explanations |               |
| **DELETE** | `/reset-chat`     | Clear chat memory           | ([GitHub][1]) |

---

## 💡 **Usage Tips**

✅ Use **thread_id** to maintain individual user sessions
✅ The agent can **query Neo4j** and **search Google**
✅ Keep **env vars** correct for Neo4j & search to avoid failures

---
