"""Agents that first use Google (SerpAPI) and then "click" around the web pages (follow urls)."""
import io
import json
import os

import httpx
import pypdf
from agentforum.forum import InteractionContext
from serpapi import GoogleSearch

from forum_versus_gaia.forum_versus_gaia_config import forum, fast_gpt_completion

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


@forum.agent(
    description=(
        "Much like a search engine but finds and returns from the internet PDFs that satisfy a search query. Useful "
        "when the information needed to answer a question is more likely to be found in some kind of PDF document "
        "rather than a webpage. Input should be a search query. (NOTE: {AGENT_ALIAS} already knows that its job "
        "is to look for PDFs, so you shouldn’t include the word “PDF” in your query.)"
    ),
)
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
