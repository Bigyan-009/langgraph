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
        "Classify the question into one category only.\n"
        "Return only one word from this list:\n"
        "cs, study, general\n\n"
        "Question: " + state["question"]
    )

    response = llm.invoke(prompt)

    category = response.content.strip().lower()

    return {"category": category}


def choose_route(state: ChatState):
    if state["category"] == "cs":
        return "cs_route"
    elif state["category"] == "study":
        return "study_route"
    else:
        return "general_route"


def cs_answer(state: ChatState):
    prompt = (
        "You are a computer science tutor. "
        "Explain the answer simply for an exam student.\n\n"
        "Question: " + state["question"]
    )

    response = llm.invoke(prompt)

    return {"answer": response.content}


def study_answer(state: ChatState):
    prompt = (
        "You are a study advisor. "
        "Give practical and simple study advice.\n\n"
        "Question: " + state["question"]
    )

    response = llm.invoke(prompt)

    return {"answer": response.content}


def general_answer(state: ChatState):
    prompt = (
        "Answer the question clearly and simply.\n\n" "Question: " + state["question"]
    )

    response = llm.invoke(prompt)

    return {"answer": response.content}


graph = StateGraph(ChatState)

graph.add_node("classify_question", classify_question)
graph.add_node("cs_answer", cs_answer)
graph.add_node("study_answer", study_answer)
graph.add_node("general_answer", general_answer)

graph.set_entry_point("classify_question")

graph.add_conditional_edges(
    "classify_question",
    choose_route,
    {
        "cs_route": "cs_answer",
        "study_route": "study_answer",
        "general_route": "general_answer",
    },
)

graph.add_edge("cs_answer", END)
graph.add_edge("study_answer", END)
graph.add_edge("general_answer", END)

app = graph.compile()

result = app.invoke(
    {
        "question": "What is the longest river in the world?",
        "category": "",
        "answer": "",
    }
)

print("Category:", result["category"])
print("Answer:")
print(result["answer"])
