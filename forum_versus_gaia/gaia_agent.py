"""
Try out a question from the GAIA dataset.
"""
from agentforum.forum import InteractionContext

from forum_versus_gaia.forum_versus_gaia_config import forum, slow_gpt_completion, fast_gpt_completion
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

MAX_NUM_OF_RESEARCHES = 2


@forum.agent
async def gaia_agent(ctx: InteractionContext, **kwargs) -> None:
    """
    A general AI assistant that can answer questions that require research.
    """
    accumulated_context = []
    for research_idx in range(MAX_NUM_OF_RESEARCHES):
        if research_idx > 0:
            ctx.respond("DOING MORE RESEARCH...")

        if accumulated_context:
            context_str = "\n\n".join([msg.content for msg in accumulated_context])
            context_msgs = pdf_finder_agent.quick_call(
                [
                    ctx.request_messages,
                    f"Information that was found so far (no need to look for it again):\n\n{context_str}",
                ]
            )
        else:
            context_msgs = pdf_finder_agent.quick_call(ctx.request_messages)

        accumulated_context.extend(await context_msgs.amaterialize_as_list())

        prompt = [
            {
                "content": GAIA_SYSTEM_PROMPT,
                "role": "system",
            },
            {
                "content": "In order to answer the question use the following info:",
                "role": "system",
            },
            *accumulated_context,
            {
                "content": "HERE GOES THE QUESTION:",
                "role": "system",
            },
            *await ctx.request_messages.amaterialize_as_list(),
        ]
        answer_msg = slow_gpt_completion(prompt=prompt, pl_tags=["FINISH"], **kwargs)
        ctx.respond(answer_msg)

        prompt = [
            {
                "content": (
                    "Your job is to judge whether the user's question was answered or not. Here is the user's "
                    "question:"
                ),
                "role": "system",
            },
            *await ctx.request_messages.amaterialize_as_list(),
            {
                "content": "And here is the answer:",
                "role": "system",
            },
            {
                "content": await answer_msg.amaterialize_content(),
                "role": "assistant",
            },
            {
                "content": (
                    "Choose a single option that best describes what happened:\n"
                    "\n"
                    "1. The question was answered.\n"
                    "2. There was not enough information in the context to answer the question.\n"
                    "3. None of the above.\n"
                    "\n"
                    "Answer with a single number and no other text.\n"
                    "\n"
                    "ANSWER:"
                ),
                "role": "system",
            },
        ]
        option_msg = fast_gpt_completion(prompt=prompt, pl_tags=["CHECK_ANSWER"])
        for char in await option_msg.amaterialize_content():
            if char.isdigit():
                if char == "1":
                    return  # the question was answered
                break  # the question was not answered


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
