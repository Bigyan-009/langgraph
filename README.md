
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
├── requirements.txt
├── .gitignore
└── README.md
```
## Files Explained

| File | Description |
|---|---|
| `category_route_graph.py` | Demonstrates question classification and route based answering |
| `chat_loop_graph.py` | Simple continuous chatbot loop with temporary memory |
| `checkpointer_chat.py` | Chatbot memory using LangGraph checkpointer |
| `mongodb_memory_chat.py` | Persistent LangGraph memory using MongoDB |
| `tool_graph.py` | Basic manual tool routing example |
| `tool_calling_graph.py` | Proper tool calling using ToolNode and tools_condition |
| `two_step_ai_graph.py` | Two step AI workflow with classification and answering |
