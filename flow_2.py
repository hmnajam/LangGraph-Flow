import operator
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_core.pydantic_v1 import BaseModel
class Queries(BaseModel):
    queries: List[str]

class AgentState(TypedDict):
    flow: List[str]
    project_status: bool
    project_name: str
    product_name: str
    
    new_version: str
    new_price: int
        
from google.colab import userdata
GOOGLE_API_KEY = userdata.get('GEMINI_API_KEY')
# print('GOOGLE_API_KEY: ', GOOGLE_API_KEY)

from langchain_google_genai import ChatGoogleGenerativeAI
# MODEL_NAME = "gemini-1.5-flash"
MODEL_NAME = "gemini-2.0-flash-exp"

model = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    generation_config={"response_mime_type": "application/json",
    "response_schema": list[AgentState]},
    temperature=0.1,
    api_key = GOOGLE_API_KEY,
    )

from tavily import TavilyClient
tavily = TavilyClient(userdata.get('Tavily_API_Key'))
# print('TAVILY_API_KEY: ',tavily.api_key)




















def ask_project(state: AgentState):
    messages = [
        SystemMessage(content=ask_project_prompt),
        HumanMessage(content=state['flow'])
    ]
    response = model.invoke(messages)
    return {"project_status": response.content}



def select_project(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=select_project_prompt),
        HumanMessage(content=state['flow'])
    ])
    return {"content": queries}




def select_product(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['flow']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=select_product_prompt.format(content=content)
        ),
        user_message
        ]
    response = model.invoke(messages)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 0) + 1
    }




def modify_prices(state: AgentState):
    messages = [
        SystemMessage(content=modify_prices_prompt),
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"new_price": response.content}





def create_version(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=ARTIST_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}




def roll_date(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=ARTIST_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}









def has_project(state: AgentState) -> str:
    # Check if project exists
    if state["project_status"]:
        print("Have project")
        return "select_project"
    else:
        print("No project")
        return "select_product"





def product_options(state: AgentState) -> str:
    # Check if project exists
    if state["modify"]:
        print("modify_prices")
        return "modify_prices"
    else:
        print("create_version")
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
    {"select_project": "select_project", "select_product": "select_product"}
)
builder.add_edge("select_project", "select_product")
builder.add_conditional_edges(
    "select_product",
    product_options,
    {"modify_prices": "modify_prices", "create_version": "create_version"}
)
builder.add_edge("create_version", "roll_date")
builder.add_edge("modify_prices",  END)
builder.add_edge("roll_date",  END)







from IPython.display import Image, display, Markdown
# memory = SqliteSaver.from_conn_string(":memory:")
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
  graph = builder.compile(checkpointer=checkpointer)
Image(graph.get_graph().draw_png())







with SqliteSaver.from_conn_string(":memory:") as checkpointer:
  graph = builder.compile(checkpointer=checkpointer)
  thread = {"configurable": {"thread_id": "1"}}

  for s in graph.stream({
    'flow': "Hi there",
  }, thread):
    # print(f"\n\nNew output starting here: \n{s}")
    # print(f"Type of s: {type(s)}")
    markdown_content = ""
    for key, value in s.items():
      markdown_content += f"\n### {key}\n"  # Print key as a heading
      if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            markdown_content += f"- **{sub_key}**: {sub_value}\n"  # Print sub-keys and values
      else:
        markdown_content += f"- {value}\n"  # Print value directly if not a dict

      # Display the Markdown content
      display(Markdown(markdown_content))