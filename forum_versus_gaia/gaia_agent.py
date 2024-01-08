"""
Try out a question from the GAIA dataset.
"""
from agentforum.forum import InteractionContext

from forum_versus_gaia.forum_versus_gaia_config import forum, slow_gpt_completion
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


@forum.agent
async def gaia_agent(ctx: InteractionContext, **kwargs) -> None:
    """
    A general AI assistant that can answer questions that require research.
    """
    context_msgs = pdf_finder_agent.quick_call(ctx.request_messages)
    prompt = [
        {
            "content": GAIA_SYSTEM_PROMPT,
            "role": "system",
        },
        {
            "content": "In order to answer the question use the following info:",
            "role": "system",
        },
        *await context_msgs.amaterialize_as_list(),
        {
            "content": "HERE GOES THE QUESTION:",
            "role": "system",
        },
        # TODO Oleksandr: should be possible to just send ctx.request_messages instead of *...
        *await ctx.request_messages.amaterialize_as_list(),
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
        "What integer-rounded percentage of the total length of the harlequin shrimp recorded in Omar "
        "Valencfia-Mendez 2017 paper was the sea star fed to the same type of shrimp in G. Curt Fiedler's 2002 "
        "paper?"
    )
    await arun_assistant(question)
