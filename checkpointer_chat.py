import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage


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

memory = MemorySaver()

app = graph.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "chat_1"}}
config2 = {"configurable": {"thread_id": "chat_2"}}


print("Chatbot started. Type 'exit' to stop.")

while True:
    print("Chat 1:")
    user_input = input("You: ")
    print("Chat 2:")
    user_input2 = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot stopped.")
        break

    result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    result2 = app.invoke(
        {"messages": [HumanMessage(content=user_input2)]}, config=config
    )

    print("AI:", result["messages"][-1].content)
    print("AI2:", result2["messages"][-1].content)
