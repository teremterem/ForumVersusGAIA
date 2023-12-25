"""
Try out a question from the GAIA dataset.
"""
import io
import json
import os

import httpx
import pypdf
from agentforum.forum import InteractionContext
from serpapi import GoogleSearch

from forum_versus_gaia.forum_versus_gaia_config import forum, fast_gpt_completion, slow_gpt_completion

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

FindPDF: Much like a search engine but finds and returns from the internet PDFs that satisfy a search query. Useful \
when the information needed to answer a question is more likely to be found in some kind of PDF document rather than \
a webpage. Input should be a search query. (NOTE: FindPDF already knows that its job is to look for PDFs, so you \
shouldn’t include the word “PDF” in your query.)

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [FindPDF]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!\
"""

EXTRACT_URL_PROMPT = """\
Your name is FindPDF. You will be provided with a SerpAPI JSON response that contains a list of search results for \
a given user query. The user is looking for a PDF document. Your job is to extract a URL that, in your opinion, \
is the most likely to contain the PDF document the user is looking for.\
"""

EXTRACT_URL_FROM_PAGE_PROMPT = """\
Your name is FindPDF. You will be provided with the content of a web page that was found via web search with a \
given user query. The user is looking for a PDF document. Your job is to extract from this web page a URL that, in \
your opinion, is the most likely to lead to the PDF document the user is looking for.\
"""


@forum.agent
async def pdf_finder_agent(ctx: InteractionContext, original_question: str = "") -> None:
    """Call SerpAPI directly."""
    query = await ctx.request_messages.amaterialize_concluding_content()
    query = query.strip()

    search = GoogleSearch(
        {
            "q": query,
            "api_key": os.environ["SERPAPI_API_KEY"],
        }
    )
    organic_results = search.get_dict()["organic_results"]

    prompt = [
        {
            "content": EXTRACT_URL_PROMPT,
            "role": "system",
        },
        {
            "content": (
                f"USER QUERY: {query}\n\nTHE ORIGINAL QUESTION THIS QUERY WAS DERIVED FROM: {original_question}"
            ),
            "role": "user",
        },
        {
            "content": f"SERPAPI SEARCH RESULTS: {json.dumps(organic_results)}",
            "role": "user",
        },
        {
            "content": "PLEASE ONLY RETURN A URL AND NO OTHER TEXT.\n\nURL:",
            "role": "system",
        },
    ]
    page_url = (await fast_gpt_completion(prompt=prompt).amaterialize_content()).strip()

    for _ in range(5):
        httpx_response = await httpx.AsyncClient().get(page_url)
        # check if mimetype is pdf
        if httpx_response.headers["content-type"] == "application/pdf":
            break

        prompt = [
            {
                "content": EXTRACT_URL_FROM_PAGE_PROMPT,
                "role": "system",
            },
            {
                "content": (
                    f"USER QUERY: {query}\n\nTHE ORIGINAL QUESTION THIS QUERY WAS DERIVED FROM: {original_question}"
                ),
                "role": "user",
            },
            {
                "content": f"PAGE CONTENT:\n\n{httpx_response.text}",
                "role": "user",
            },
            {
                "content": "PLEASE ONLY RETURN A URL AND NO OTHER TEXT.\n\nURL:",
                "role": "system",
            },
        ]
        page_url = (await fast_gpt_completion(prompt=prompt).amaterialize_content()).strip()
    else:
        raise RuntimeError("Could not find a PDF document.")  # TODO Oleksandr: custom exception ?

    pdf_reader = pypdf.PdfReader(io.BytesIO(httpx_response.content))
    pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
    ctx.respond(pdf_text)


@forum.agent
async def gaia_agent(ctx: InteractionContext, **kwargs) -> None:
    """An agent that uses OpenAI ChatGPT under the hood. It sends the full chat history to the OpenAI API."""
    # TODO Oleksandr: find a prompt format that allows full chat history to be passed
    last_message = await ctx.request_messages.amaterialize_concluding_content()
    last_message = last_message.strip()

    prompt = [
        {
            "content": CHOOSE_TOOL_PROMPT,
            "role": "system",
        },
        {
            "content": f"Question: {last_message}\nThought:",
            "role": "user",
        },
    ]
    query_msg_content = await fast_gpt_completion(prompt=prompt, stop="\nObservation:").amaterialize_content()
    query = query_msg_content.split("Action Input:")[1].strip()

    pdf_content = await pdf_finder_agent.quick_call(
        # TODO Oleksandr: introduce "reply_to" feature in the message tree and use it instead of agent kwarg ?
        query,
        original_question=last_message,
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
        "In Series 9, Episode 11 of Doctor Who, the Doctor is trapped inside an ever-shifting maze. What is this "
        "location called in the official script for the episode? Give the setting exactly as it appears in the "
        "first scene heading."
    )
    await run_assistant(question)
