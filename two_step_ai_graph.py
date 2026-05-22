import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


load_dotenv()


class ChatState(TypedDict):
    question: str
    category: str
    answer: str


llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))


def classify_question(state: ChatState):
    prompt = (
        "Classify this question into one category only. "
        "Choose from: computer science, general knowledge, study advice, other.\n\n"
        "Question: " + state["question"]
    )

    response = llm.invoke(prompt)

    return {"category": response.content}


def answer_question(state: ChatState):
    prompt = (
        "The question category is: " + state["category"] + "\n\n"
        "Answer the question clearly and simply.\n\n"
        "Question: " + state["question"]
    )

    response = llm.invoke(prompt)

    return {"answer": response.content}


graph = StateGraph(ChatState)

graph.add_node("classify_question", classify_question)
graph.add_node("answer_question", answer_question)

graph.set_entry_point("classify_question")

graph.add_edge("classify_question", "answer_question")
graph.add_edge("answer_question", END)

app = graph.compile()

result = app.invoke({"question": "What is Ram?", "category": "", "answer": ""})

print("Category:")
print(result["category"])

print("\nAnswer:")
print(result["answer"])
