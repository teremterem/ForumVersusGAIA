"""
Try out a question from the GAIA dataset.
"""
from agentforum.forum import InteractionContext

from forum_versus_gaia.forum_versus_gaia_config import forum, fast_gpt_completion, slow_gpt_completion
from forum_versus_gaia.more_agents.pdf_finder_agent import pdf_finder_agent

GAIA_SYSTEM_PROMPT = """\
You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the \
following template: FINAL ANSWER: [YOUR FINAL ANSWER].
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign \
unless specified otherwise.
If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in \
plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the \
list is a number or a string.\
"""

CHOOSE_TOOL_PROMPT = """\
Answer the following questions as best you can. You have access to the following tools:

{AGENT_DESCRIPTIONS}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{AGENT_NAMES}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!\
"""


@forum.agent
async def gaia_agent(ctx: InteractionContext, **kwargs) -> None:
    """
    A general AI assistant that can answer questions that require research.
    """
    agents = [
        # browsing_agent,
        pdf_finder_agent,
    ]

    agent_names = ", ".join(agent.alias for agent in agents)
    agent_descriptions = "\n".join(f"{agent.alias}: {agent.description}" for agent in agents)

    # TODO Oleksandr: find a prompt format that allows full chat history to be passed ?
    question = await ctx.request_messages.amaterialize_concluding_content()
    question = question.strip()

    prompt = [
        {
            "content": CHOOSE_TOOL_PROMPT.format(AGENT_NAMES=agent_names, AGENT_DESCRIPTIONS=agent_descriptions),
            "role": "system",
        },
        {
            "content": f"Question: {question}\nThought:",
            "role": "user",
        },
    ]
    query_msg_content = await fast_gpt_completion(
        prompt=prompt, stop="\nObservation:", pl_tags=["START"]
    ).amaterialize_content()
    query = query_msg_content.split("Action Input:")[1].strip()

    content = await pdf_finder_agent.quick_call(
        query,
        # TODO Oleksandr: `branch_from` should accept either a message promise or a concrete message or a message id
        #  or even a message sequence (but not your own list of messages ?)
        branch_from=await ctx.request_messages.aget_concluding_msg_promise(),
    ).amaterialize_concluding_content()

    prompt = [
        {
            "content": GAIA_SYSTEM_PROMPT,
            "role": "system",
        },
        {
            "content": "In order to answer the question use the following info:",
            "role": "system",
        },
        {
            "content": content,
            "role": "user",
        },
        {
            "content": "HERE GOES THE QUESTION:",
            "role": "system",
        },
        # TODO Oleksandr: should be possible to just send ctx.request_messages instead of *...
        *await ctx.request_messages.amaterialize_full_history(),
    ]
    ctx.respond(slow_gpt_completion(prompt=prompt, pl_tags=["FINISH"], **kwargs))


async def arun_assistant(question: str) -> str:
    """Run the assistant. Return the final answer in upper case."""
    print("\n\nQUESTION:", question)

    assistant_responses = gaia_agent.quick_call(question, stream=True)

    async for response in assistant_responses:
        print("\n\033[36;1mGPT: ", end="", flush=True)
        async for token in response:
            print(token.text, end="", flush=True)
        print("\033[0m")
    print()

    final_answer = await assistant_responses.amaterialize_concluding_content()
    final_answer = final_answer.upper()
    final_answer = final_answer.split("FINAL ANSWER:")[1].strip()
    return final_answer


async def amain() -> None:
    """
    Run the assistant on a question from the GAIA dataset.
    """
    question = (
        "On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. This article "
        "mentions a team that produced a paper about their observations, linked at the bottom of the article. "
        "Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?"
    )
    await arun_assistant(question)
