"""Agents that first use Google (SerpAPI) and then "click" around the web pages (follow urls)."""
import io
import json
import os
from typing import Any
from urllib.parse import urlparse

import httpx
import pypdf
from agentforum.forum import InteractionContext
from serpapi import GoogleSearch

from forum_versus_gaia.forum_versus_gaia_config import forum, fast_gpt_completion, REMOVE_GAIA_LINKS
from forum_versus_gaia.utils import NotAUrlError, render_conversation

EXTRACT_PDF_URL_PROMPT = """\
Your name is {AGENT_ALIAS}. You will be provided with a SerpAPI JSON response that contains a list of search results \
for a given user query. The user is looking for a PDF document. Your job is to extract a URL that, in your opinion, \
is the most likely to contain the PDF document the user is looking for.\
"""

EXTRACT_URL_PROMPT = """\
Your name is {AGENT_ALIAS}. You will be provided with a SerpAPI JSON response that contains a list of search results \
for a given user query. The user is looking for an answer to the ORIGINAL QUESTION (you will se it below). Your job \
is to extract a URL that, in your opinion, is the most likely to contain the answer the user is looking for.\
"""

EXTRACT_PDF_URL_FROM_PAGE_PROMPT = """\
Your name is {AGENT_ALIAS}. You will be provided with the content of a web page that was found via web search with a \
given user query. The user is looking for a PDF document. Your job is to extract from this web page a URL that, in \
your opinion, is the most likely to lead to the PDF document the user is looking for.\
"""


@forum.agent
async def pdf_finder_agent(ctx: InteractionContext) -> None:
    """
    Much like a search engine but finds and returns from the internet PDFs that satisfy a search query. Useful when
    the information needed to answer a question is more likely to be found in some kind of PDF document rather than
    a webpage. Input should be a search query. (NOTE: {AGENT_ALIAS} already knows that its job is to look for PDFs,
    so you shouldn’t include the word “PDF” in your query.)
    """
    full_conversation = await ctx.request_messages.amaterialize_full_history()
    full_conversation_str = render_conversation(full_conversation)
    query = full_conversation[-1].content.strip()

    organic_results = get_serpapi_results(query)

    prompt = [
        {
            "content": EXTRACT_PDF_URL_PROMPT.format(AGENT_ALIAS=ctx.this_agent.alias),
            "role": "system",
        },
        {
            "content": full_conversation_str,
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
    assert_valid_url(page_url)

    # TODO TODO TODO

    async with get_httpx_client() as httpx_client:
        for _ in range(5):
            httpx_response = await httpx_client.get(page_url)
            # check if mimetype is pdf
            if httpx_response.headers["content-type"] == "application/pdf":
                break

            prompt = [
                {
                    "content": EXTRACT_PDF_URL_FROM_PAGE_PROMPT.format(AGENT_ALIAS=ctx.this_agent.alias),
                    "role": "system",
                },
                {
                    "content": full_conversation_str,
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
            assert_valid_url(page_url)
        else:
            raise RuntimeError("Could not find a PDF document.")  # TODO Oleksandr: custom exception ?

    pdf_reader = pypdf.PdfReader(io.BytesIO(httpx_response.content))
    pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
    ctx.respond(pdf_text)


@forum.agent
async def browsing_agent(ctx: InteractionContext, original_question: str = "") -> None:
    """
    Using a search engine finds and returns from the internet a web page that satisfies a search query. Input should
    be a search query.
    """
    query = await ctx.request_messages.amaterialize_concluding_content()
    query = query.strip()

    organic_results = get_serpapi_results(query)

    prompt = [
        {
            "content": EXTRACT_URL_PROMPT.format(AGENT_ALIAS=ctx.this_agent.alias),
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
    assert_valid_url(page_url)

    async with get_httpx_client() as httpx_client:
        httpx_response = await httpx_client.get(page_url)

    ctx.respond(httpx_response.text)


def assert_valid_url(url: str) -> None:
    """
    Raises an exception if the given URL is not valid.
    """
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            raise ValueError
    except ValueError as exc:
        raise NotAUrlError(url) from exc


def get_httpx_client() -> httpx.AsyncClient:
    """
    Returns a httpx client with the settings we want.
    """
    return httpx.AsyncClient(follow_redirects=True, verify=False)


def get_serpapi_results(query: str, remove_gaia_links: bool = REMOVE_GAIA_LINKS) -> list[dict[str, Any]]:
    """
    Returns a list of organic results from SerpAPI for a given query.
    """
    # TODO Oleksandr: make this function async by replacing SerpAPI python client with plain aiohttp
    search = GoogleSearch(
        {
            "q": query,
            "api_key": os.environ["SERPAPI_API_KEY"],
        }
    )
    organic_results = search.get_dict()["organic_results"]
    if remove_gaia_links:
        # we don't want the agents to look up answers in the GAIA benchmark itself
        organic_results = [
            organic_result
            for organic_result in organic_results
            if "gaia-benchmark" not in organic_result["link"].lower() and "2311.12983" not in organic_result["link"]
        ]
    return organic_results
