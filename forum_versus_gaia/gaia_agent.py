"""
Try out a question from the GAIA dataset.
"""
from agentforum.forum import InteractionContext

from forum_versus_gaia.forum_versus_gaia_config import forum, fast_gpt_completion, slow_gpt_completion
from forum_versus_gaia.more_agents.browsing_agents import pdf_finder_agent

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
    """An agent that uses OpenAI ChatGPT under the hood. It sends the full chat history to the OpenAI API."""
    agents = [pdf_finder_agent]

    agent_names = ", ".join(agent.agent_alias for agent in agents)
    agent_descriptions = "\n".join(f"{agent.agent_alias}: {agent.agent_description}" for agent in agents)

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
    query_msg_content = await fast_gpt_completion(prompt=prompt, stop="\nObservation:").amaterialize_content()
    query = query_msg_content.split("Action Input:")[1].strip()

    pdf_content = await pdf_finder_agent.quick_call(
        # TODO Oleksandr: introduce "reply_to" feature in the message tree and use it instead of agent kwarg ?
        query,
        original_question=question,
    ).amaterialize_concluding_content()

    prompt = [
        {
            "content": GAIA_SYSTEM_PROMPT,
            "role": "system",
        },
        {
            "content": "In order to answer the question use the following content of a PDF document:",
            "role": "system",
        },
        {
            "content": pdf_content,
            "role": "user",
        },
        {
            "content": "HERE GOES THE QUESTION:",
            "role": "system",
        },
        # TODO Oleksandr: should be possible to just send ctx.request_messages instead of *...
        *await ctx.request_messages.amaterialize_full_history(),
    ]
    ctx.respond(slow_gpt_completion(prompt=prompt, **kwargs))


async def run_assistant(question: str) -> str:
    """Run the assistant. Return the final answer in upper case."""
    print("\n\nQUESTION:", question)

    assistant_responses = gaia_agent.quick_call(question, stream=True)

    async for response in assistant_responses:
        print("\n\033[1m\033[36mGPT: ", end="", flush=True)
        async for token in response:
            print(token.text, end="", flush=True)
        print("\033[0m")
    print()

    final_answer = await assistant_responses.amaterialize_concluding_content()
    final_answer = final_answer.upper()
    final_answer = final_answer.split("FINAL ANSWER:")[1].strip()
    return final_answer


async def main() -> None:
    """
    Run the assistant on a question from the GAIA dataset.
    """
    question = (
        "What was the volume in m^3 of the fish bag that was calculated in the University of Leicester paper "
        '"Can Hiccup Supply Enough Fish to Maintain a Dragon’s Diet?"'
    )
    await run_assistant(question)
