"""Langchain based experiments."""
from agentforum.forum import InteractionContext
from langchain import hub
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.tools.render import render_text_description


async def pdf_finder_agent(ctx: InteractionContext) -> None:
    """ReAct"""
    llm = PromptLayerChatOpenAI(temperature=0, model_name="gpt-4-1106-preview")
    tools = load_tools(["serpapi"], llm=llm)
    prompt = hub.pull("hwchase17/react")
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    llm_with_stop = llm.bind(stop=["\nObservation"])
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | llm_with_stop
        | ReActSingleInputOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)
    agent_executor.invoke(
        {
            "input": (await ctx.request_messages.amaterialize_concluding_message()).content,
        }
    )
