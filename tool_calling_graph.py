import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def calculator(expression: str) -> str:
    """Calculate a simple maths expression. Example: 45 * 29."""
    try:
        result = eval(expression)
        return str(result)
    except Exception:
        return "Calculation error"


tools = [calculator]


llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

llm_with_tools = llm.bind_tools(tools)


def assistant_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


graph = StateGraph(AgentState)

graph.add_node("assistant", assistant_node)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("assistant")

graph.add_conditional_edges("assistant", tools_condition)

graph.add_edge("tools", "assistant")

app = graph.compile()


result = app.invoke({"messages": [HumanMessage(content="Calculate 45 * 29")]})

for message in result["messages"]:
    print(type(message).__name__, ":", message.content)
