# import operator
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ChatMessage
from langchain_core.messages import AIMessage


def main():
    print("Hello from simple-langgraph-flow!")


class Queries(BaseModel):
    queries: List[str]


class AgentState(TypedDict):
    flow: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


from google.colab import userdata

GOOGLE_API_KEY = userdata.get("GEMINI_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI

MODEL_NAME = "gemini-2.0-flash"

model = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    generation_config={
        "response_mime_type": "application/json",
        "response_schema": list[AgentState],
    },
    temperature=0.1,
    api_key=GOOGLE_API_KEY,
)

from tavily import TavilyClient

tavily = TavilyClient(userdata.get("Tavily_API_Key"))
# print('TAVILY_API_KEY: ',tavily.api_key)


# You are an expert ________. Your goal is to ______
PRODUCT_CHECKING_PROMPT = """You are an expert blogger and SEO expert. Your goal is to write a high level outline of a blog. \
Write such an outline for the user provided topic. Give an outline of the blog along with any relevant SEO tips\
and instructions for the sections. Also suggest how long should be each section and how long should the blog be."""


RESEARCHER_PROMPT = """You are an expert research assistant tasked with gathering relevant information for a blog post. \
You will focus on generating research that will help in writing SEO optimized article.
Please generate a maximum of three focused search queries that will yield valuable insights and data for the topic at hand.
"""


WRITER_PROMPT = """You are an expert blogger tasked with writing an excellent blog.\
Generate the best and most SEO optimized blog possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed:

------

{content}"""


EDITOR_PROMPT = """You are an experienced editor of a blogging platform grading a blog submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""


ARTIST_PROMPT = """You are an expert artist. Your goal is to search and return artwork most related to the blog post. \
Generate a list of search queries that will gather any relevant imagery. Only generate 2 queries max."""


def project_checking_node(state: AgentState):
    messages = [
        SystemMessage(content=PRODUCT_CHECKING_PROMPT),
        HumanMessage(content=state["flow"]),
    ]
    response = model.invoke(messages)
    return {"plan": response[0].content}


def project_selection_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [SystemMessage(content=RESEARCHER_PROMPT), HumanMessage(content=state["flow"])]
    )
    content = state.get("content", [])
    for q in queries.queries:
        response = tavily.search(query=q, max_results=3)
        for r in response["results"]:
            content.append(r["content"])
    return {"content": content}


def product_selection_node(state: AgentState):
    content = "\n\n".join(state["content"] or [])
    user_message = HumanMessage(
        content=f"{state['flow']}\n\nHere is my plan:\n\n{state['plan']}"
    )
    messages = [
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        user_message,
    ]
    response = model.invoke(messages)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 0) + 1,
    }


def select_action():
    # Logic to determine action based on user input
    # Return "modify_price" or "create_version"
    return "modify_price"  # or "create_version" based on user choice


def modify_prices_node(state: AgentState):
    messages = [
        SystemMessage(content=EDITOR_PROMPT),
        HumanMessage(content=state["draft"]),
    ]
    response = model.invoke(messages)
    return {"critique": response.content}


def has_project(state):
    if state["revision_number"] > state["max_revisions"]:
        print("Reached max revisions")
        return "project_select"  # Return "project_select" when condition is true
    print("Should continue")
    return "product_select"  # Return "product_select" otherwise


def get_rollout_date_node(state: AgentState):
    messages = [
        SystemMessage(content=EDITOR_PROMPT),
        HumanMessage(content=state["draft"]),
    ]
    response = model.invoke(messages)
    return {"critique": response.content}


def create_version_node(state: AgentState):
    messages = [
        SystemMessage(content=EDITOR_PROMPT),
        HumanMessage(content=state["draft"]),
    ]
    response = model.invoke(messages)
    return {"critique": response.content}

    builder = StateGraph(AgentState)


# Adding nodes for the flow
builder.add_node("project_check", project_checking_node)
builder.add_node("project_select", project_selection_node)
builder.add_node("product_select", product_selection_node)
builder.add_node("modify_price", modify_prices_node)
builder.add_node("get_rollout_date", get_rollout_date_node)
builder.add_node("create_version", create_version_node)


def has_project(state):
    if True:
        print("Reached max revisions")
        return "project_select"  # Corrected node name
    print("Should continue")
    return "product_select"  # Corrected node name


# # Set the starting node
# builder.set_entry_point("project_check")

# # From "project_check", use a conditional edge:
# # If user has a project, go to "project_select"; if not, skip directly to "product_select"
# builder.add_conditional_edges(
#     "project_check",
#     has_project,  # Function that returns "yes" if project exists, otherwise "no"
#     {"yes": "project_select", "no": "product_select"}
# )

# # After project selection, continue to product selection (for users with a project)
# builder.add_edge("project_select", "product_select")

# # Proceed from product selection to modifying the price
# builder.add_edge("product_select", "modify_price")

# # End the flow after modifying the price
# builder.add_edge("modify_price", END)


# Set the starting node
builder.set_entry_point("project_check")

# From "project_check", use a conditional edge:
# If user has a project, go to "project_select"; if not, skip directly to "product_select"
builder.add_conditional_edges(
    "project_check",
    has_project,  # Function that returns "yes" if project exists, otherwise "no"
    {"yes": "project_select", "no": "product_select"},
)

# After project selection, continue to product selection (for users with a project)
builder.add_edge("project_select", "product_select")

# # From product selection, give two options: Modify Price or Create New Product Version
# builder.add_conditional_edges(
#     "product_select",
#     select_action,  # Function to choose between modifying price or creating a new version
#     # {"modify_price": "modify_price", "create_version": "get_rollout_date"}
#      {"project_select": "project_select", "product_select": "product_select"} # Correct the conditional edges
# )

builder.add_conditional_edges(
    "product_select",
    select_action,
    {"modify_price": "modify_price", "create_version": "get_rollout_date"},
)


# Flow for modifying the price
builder.add_edge("modify_price", END)

# Flow for creating a new product version
builder.add_edge("get_rollout_date", "create_version")
builder.add_edge("create_version", END)


from IPython.display import Image, display, Markdown

# memory = SqliteSaver.from_conn_string(":memory:")
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
Image(graph.get_graph().draw_png())


with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    thread = {"configurable": {"thread_id": "1"}}

    for s in graph.stream(
        {
            "flow": "Hi",
        },
        thread,
    ):
        # print(f"\n\nNew output starting here: \n{s}")
        # print(f"Type of s: {type(s)}")
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


if __name__ == "__main__":
    main()
