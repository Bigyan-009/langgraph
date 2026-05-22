import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.mongodb import MongoDBSaver


load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))


def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])

    return {"messages": [response]}


graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.set_entry_point("chat_node")
graph.add_edge("chat_node", END)


mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = MongoClient(mongodb_uri)

checkpointer = MongoDBSaver(client=client, db_name="langgraph_memory")

app = graph.compile(checkpointer=checkpointer)


def print_thread_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = app.get_state(config)
    messages = snapshot.values.get("messages", [])

    print("\n--- Saved messages for", thread_id, "---")

    for message in messages:
        print(type(message).__name__, ":", message.content)

    print("--------------------------------\n")


user_id = input("Enter user ID: ")

config = {"configurable": {"thread_id": user_id}}

print("Chat started. Type 'exit' to stop.")
print("Type 'show memory' to view saved messages.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chat stopped.")
        break

    if user_input.lower() == "show memory":
        print_thread_state(user_id)
        continue

    result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

    ai_reply = result["messages"][-1].content

    print("AI:", ai_reply)
