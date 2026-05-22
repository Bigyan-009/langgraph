import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph
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
    except Exception as error:
        return f"Calculation error: {error}"


@tool
def word_counter(text: str) -> str:
    """Count how many words are in the given text."""
    words = text.split()
    return str(len(words))


@tool
def uppercase_text(text: str) -> str:
    """Convert the given text into uppercase letters."""
    return text.upper()


tavily_search = TavilySearch(max_results=3, topic="general", include_answer=True)


tools = [calculator, word_counter, uppercase_text, tavily_search]


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


while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chat stopped.")
        break

    result = app.invoke({"messages": [HumanMessage(content=user_input)]})

    print("\nAI:", result["messages"][-1].content)

    print("\n--- Message Debug ---")
    for index, message in enumerate(result["messages"], start=1):
        print("\nMessage:", index)
        print("Type:", type(message).__name__)
        print("Content:", message.content)

        if hasattr(message, "tool_calls"):
            print("Tool calls:", message.tool_calls)

        if hasattr(message, "name"):
            print("Tool name:", message.name)

        if hasattr(message, "tool_call_id"):
            print("Tool call ID:", message.tool_call_id)

    print("\n---------------------\n")
