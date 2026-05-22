import os
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


load_dotenv()


class ChatState(TypedDict):
    messages: List[str]
    answer: str


llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))


def chat_node(state: ChatState):
    conversation = "\n".join(state["messages"])

    response = llm.invoke(conversation)

    return {"answer": response.content}


graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.set_entry_point("chat_node")
graph.add_edge("chat_node", END)

app = graph.compile()


memory = []

print("Chatbot started. Type 'exit' to stop.")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot stopped.")
        break

    memory.append("User: " + user_input)

    result = app.invoke({"messages": memory, "answer": ""})

    ai_answer = result["answer"]

    print("AI:", ai_answer)

    memory.append("AI: " + ai_answer)
