import os
import uuid
from typing import Annotated, Any, List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict

from langchain_ai21.chat.chat_maestro import ChatMaestro

os.environ["OPENAI_API_KEY"] = "openai_api_key"
os.environ["AI21_API_KEY"] = "ai21_api_key"
os.environ["AI21_API_HOST"] = "ai21_api_host"

template = """Your job is to get information from a user about what type of email
 template they want to create.

You should get the following information from them:

- What the objective of the email is
- What variables will be passed into the email
- Any requirements that the output MUST adhere to

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly
 guess.

After you are able to discern all the information, call the relevant tool."""


def get_messages_info(messages: list[SystemMessage]) -> List[SystemMessage]:
    return [SystemMessage(content=template)] + messages


class EmailInstructions(BaseModel):
    """Instructions on how to generate the email template."""

    objective: str
    variables: List[str]
    requirements: List[str]


llm_prompt = ChatMaestro()
llm_info = ChatOpenAI(temperature=0)
llm_with_tool = llm_info.bind_tools([EmailInstructions])


def info_chain(state: dict) -> dict:
    messages = get_messages_info(state["messages"])
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}


# Function to get the messages for the prompt
# Will only get messages AFTER the tool call
def get_prompt_messages_maestro(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], dict[str, Any]]:
    tool_call = {}
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
        elif tool_call is not None:
            other_msgs.append(m)
    return other_msgs, tool_call


def prompt_gen_chain(state: dict) -> dict:
    messages, tool_call = get_prompt_messages_maestro(state["messages"])
    objective = tool_call.pop("objective")
    variables = tool_call.get("variables")
    maestro_input = f"generate a prompt that meets the following objective: {objective}"

    if variables:
        maestro_input += f" with the following variables: {variables}"

    response = llm_prompt.invoke([SystemMessage(maestro_input)], **tool_call)
    return {"messages": [response]}


def get_state(state: dict) -> str:
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"


class State(TypedDict):
    messages: Annotated[list, add_messages]


memory = MemorySaver()
workflow = StateGraph(State)
workflow.add_node("info", info_chain)
workflow.add_node("prompt", prompt_gen_chain)


@workflow.add_node
def add_tool_message(state: State) -> dict:
    return {
        "messages": [
            ToolMessage(
                content="Generating email template...",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        ]
    }


workflow.add_conditional_edges("info", get_state, ["add_tool_message", "info", END])
workflow.add_edge("add_tool_message", "prompt")
workflow.add_edge("prompt", END)
workflow.add_edge(START, "info")
graph = workflow.compile(checkpointer=memory)

cached_human_responses = [
    "hi!",
    "write a prompt",
    "1 email to reach out to sell maestro the next ai tool, 2 recipient_first_name,"
    " recipient_last_name, 3 the email should contain 5 rows",
    "Sritala Hollinger",
    "q",
]

cached_response_index = 0
config: RunnableConfig = {
    "configurable": {
        "thread_id": str(uuid.uuid4()),
    }
}

while True:
    try:
        user = input("User (q/Q to quit): ")
    except EOFError:
        user = cached_human_responses[cached_response_index]
        cached_response_index += 1
    print(f"User (q/Q to quit): {user}")
    if user in {"q", "Q"}:
        print("AI: Bye bye")
        break
    output = None
    for output in graph.stream(
        {"messages": [HumanMessage(content=user)]}, config=config, stream_mode="updates"
    ):
        last_message = next(iter(output.values()))["messages"][-1]
        last_message.pretty_print()

    if output and "prompt" in output:
        print("Done!")
