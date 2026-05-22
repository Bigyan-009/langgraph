import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


load_dotenv()


class AgentState(TypedDict):
    question: str
    route: str
    answer: str


llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))


def decide_route(state: AgentState):
    question = state["question"].lower()

    if any(symbol in question for symbol in ["+", "-", "*", "/", "calculate"]):
        return {"route": "calculator"}

    return {"route": "ai"}


def choose_next_node(state: AgentState):
    if state["route"] == "calculator":
        return "calculator_route"
    else:
        return "ai_route"


def calculator_tool(state: AgentState):
    question = state["question"]

    expression = (
        question.replace("calculate", "")
        .replace("what is", "")
        .replace("?", "")
        .strip()
    )

    try:
        result = eval(expression)

        return {"answer": "The answer is " + str(result)}

    except Exception:
        return {"answer": "Sorry, I could not calculate that."}


def ai_answer(state: AgentState):
    response = llm.invoke(state["question"])

    return {"answer": response.content}


graph = StateGraph(AgentState)

graph.add_node("decide_route", decide_route)
graph.add_node("calculator_tool", calculator_tool)
graph.add_node("ai_answer", ai_answer)

graph.set_entry_point("decide_route")

graph.add_conditional_edges(
    "decide_route",
    choose_next_node,
    {"calculator_route": "calculator_tool", "ai_route": "ai_answer"},
)

graph.add_edge("calculator_tool", END)
graph.add_edge("ai_answer", END)

app = graph.compile()

result = app.invoke(
    {
        "question": "who is the president of the united states? and calculate",
        "route": "",
        "answer": "",
    }
)

print("Route:", result["route"])
print("Answer:", result["answer"])
