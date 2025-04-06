import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel
from IPython.display import display, Markdown
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import (
    ask_project_prompt,
    select_project_prompt,
    select_product_prompt,
    modify_prices_prompt,
    create_version_prompt,
    roll_date_prompt,
)


class Queries(BaseModel):
    queries: List[str]


class AgentState(TypedDict):
    flow: List[str]
    project_status: bool
    project_name: str
    product_name: str

    new_version: str
    new_price: int
    rollout_date: str
    modify: bool


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.0-flash-exp"
model = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    generation_config={
        "response_mime_type": "application/json",
        "response_schema": list[AgentState],
    },
    temperature=0.1,
    api_key=GOOGLE_API_KEY,
)


def ask_project(state: AgentState):
    messages = [
        SystemMessage(content=ask_project_prompt),
        HumanMessage(content=state["flow"]),
    ]
    response = model.invoke(messages)
    return {"project_status": response.content}


def select_project(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=select_project_prompt),
            HumanMessage(content=state["flow"]),
        ]
    )
    return {"content": queries}


def select_product(state: AgentState):
    content = "\n\n".join(state["content"] or [])
    user_message = HumanMessage(
        content=f"{state['flow']}\n\nHere is the product:\n\n{state['product_name']}"
    )
    messages = [
        SystemMessage(content=select_product_prompt.format(content=content)),
        user_message,
    ]
    response = model.invoke(messages)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 0) + 1,
    }


def modify_prices(state: AgentState):
    messages = [
        SystemMessage(content=modify_prices_prompt),
        HumanMessage(content=state["draft"]),
    ]
    response = model.invoke(messages)
    return {"new_price": response.content}


def create_version(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=create_version_prompt),
            HumanMessage(content=state["version_number"]),
        ]
    )
    content = state["flow"] or []
    for q in queries.queries:
        content.append(q)
    return {"content": content}


def roll_date(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=roll_date_prompt),
            HumanMessage(content=state["rollout_date"]),
        ]
    )
    content = state["content"] or []
    for q in queries.queries:
        content.append(q)
    return {"content": content}


def has_project(state: AgentState) -> str:
    if state.get("project_status"):
        return "select_project"
    return "select_product"


def product_options(state: AgentState) -> str:
    if state.get("modify"):
        return "modify_prices"
    return "create_version"



builder = StateGraph(AgentState)

builder.add_node("ask_project", ask_project)
builder.add_node("select_project", select_project)
builder.add_node("select_product", select_product)
builder.add_node("modify_prices", modify_prices)
builder.add_node("create_version", create_version)
builder.add_node("roll_date", roll_date)


builder.set_entry_point("ask_project")

builder.add_conditional_edges(
    "ask_project",
    has_project,
    {"select_project": "select_project", "select_product": "select_product"},
)
builder.add_edge("select_project", "select_product")
builder.add_conditional_edges(
    "select_product",
    product_options,
    {"modify_prices": "modify_prices", "create_version": "create_version"},
)
builder.add_edge("create_version", "roll_date")
builder.add_edge("modify_prices", END)
builder.add_edge("roll_date", END)


with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    thread = {"configurable": {"thread_id": "1"}}

    for s in graph.stream(
        {
            "flow": "Hi there",
        },
        thread,
    ):
        markdown_content = ""
        for key, value in s.items():
            markdown_content += f"\n### {key}\n"  # Print key as a heading
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    markdown_content += (
                        f"- **{sub_key}**: {sub_value}\n"  # Print sub-keys and values
                    )
            else:
                markdown_content += f"- {value}\n"  # Print value directly if not a dict

            # Display the Markdown content
            display(Markdown(markdown_content))
