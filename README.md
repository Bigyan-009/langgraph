# LangGraph Practice

This repository contains my beginner practice work for learning LangGraph step by step. The examples cover basic graph structure, state management, conditional routing, LLM integration, tool use, chat memory, checkpointers, and MongoDB based persistent memory.

## Topics Covered

- Basic LangGraph state, nodes and edges
- Multiple node workflows
- Conditional edges and routing
- LLM integration with ChatOpenAI
- Category based question routing
- Simple tool use
- Proper tool calling with ToolNode
- Chat memory using message state
- Checkpointer memory
- MongoDB based persistent memory
- Thread based conversation separation

## Project Structure

```
agents/
├── category_route_graph.py
├── chat_loop_graph.py
├── checkpointer_chat.py
├── mongodb_memory_chat.py
├── tool_calling_graph.py
├── tool_graph.py
├── two_step_ai_graph.py
├── multiple_tools_graph.py
├── tool_calling_mongodb_graph.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Files Explained

| File                            | Description                                                                                                             |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `category_route_graph.py`       | Demonstrates question classification and route based answering                                                          |
| `chat_loop_graph.py`            | Simple continuous chatbot loop with temporary memory                                                                    |
| `checkpointer_chat.py`          | Chatbot memory using LangGraph checkpointer                                                                             |
| `mongodb_memory_chat.py`        | Persistent LangGraph memory using MongoDB                                                                               |
| `tool_graph.py`                 | Basic manual tool routing example                                                                                       |
| `tool_calling_graph.py`         | Proper tool calling using ToolNode and tools_condition                                                                  |
| `two_step_ai_graph.py`          | Two step AI workflow with classification and answering                                                                  |
| `multiple_tools_graph.py`       | Multiple tools agent using calculator, word counter, uppercase converter, and Tavily web search                         |
| `tool_calling_mongodb_graph.py` | Combines LangGraph tool calling with MongoDBSaver so the agent can use tools and remember conversations across sessions |

## Technologies Used

- Python
- LangGraph
- LangChain
- OpenAI API
- MongoDB
- PyMongo
- Tavily Search
- Python Dotenv

## Installation

Clone the repository:

```bash
git clone https://github.com/Bigyan-009/langgraph.git
cd langgraph
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows PowerShell:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
MONGODB_URI=mongodb://localhost:27017
TAVILY_API_KEY=your_tavily_api_key_here
```

Do not upload the `.env` file to GitHub.

## Running the Examples

Run any file using:

```bash
python filename.py
```

Examples:

```bash
python tool_calling_graph.py
python multiple_tools_graph.py
python tool_calling_mongodb_graph.py
```

## Main Learning Concepts

- `StateGraph` is used to create the graph workflow.
- `TypedDict` defines the structure of the graph state.
- Nodes are Python functions that receive and update state.
- Edges control the movement between nodes.
- Conditional edges allow the graph to choose different paths.
- `add_messages` appends new chat messages to previous messages.
- `ToolNode` executes tools requested by the LLM.
- `tools_condition` checks whether the latest AI message contains tool calls.
- `MongoDBSaver` stores LangGraph checkpoints in MongoDB.
- `thread_id` separates conversations between users.

## Notes

This repository is for learning and practice. Some tools use simple beginner-friendly code. For production use, tools should be validated carefully, API keys should be kept secure, and database storage should be designed properly.

## Author

Bigyan Dahal
