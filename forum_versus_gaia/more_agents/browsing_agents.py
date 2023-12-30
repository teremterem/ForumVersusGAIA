"""Agents that first use Google (SerpAPI) and then "click" around the web pages (follow urls)."""
import io
import json

import pypdf
from agentforum.forum import InteractionContext, ConversationTracker

from forum_versus_gaia.forum_versus_gaia_config import forum, fast_gpt_completion
from forum_versus_gaia.utils import (
    render_conversation,
    get_serpapi_results,
    assert_valid_url,
    get_httpx_client,
    is_valid_url,
)

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
async def pdf_finder_agent(ctx: InteractionContext, recursion: int = 5) -> None:
    """
    Much like a search engine but finds and returns from the internet PDFs that satisfy a search query. Useful when
    the information needed to answer a question is more likely to be found in some kind of PDF document rather than
    a webpage. Input should be a search query. (NOTE: {AGENT_ALIAS} already knows that its job is to look for PDFs,
    so you shouldn’t include the word “PDF” in your query.)
    """
    if recursion <= 0:
        # TODO Oleksandr: custom exception ?
        raise RuntimeError("I couldn't find a PDF document within a reasonable number of hops.")

    full_conversation = await ctx.request_messages.amaterialize_full_history()
    query_or_url = full_conversation[-1].content.strip()

    if is_valid_url(query_or_url):
        async with get_httpx_client() as httpx_client:
            httpx_response = await httpx_client.get(query_or_url)

        if httpx_response.headers["content-type"] == "application/pdf":
            # pdf was found! returning its text
            pdf_reader = pypdf.PdfReader(io.BytesIO(httpx_response.content))
            pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            ctx.respond(pdf_text)
            return

        prompt_header = EXTRACT_PDF_URL_FROM_PAGE_PROMPT
        prompt_context = f"PAGE CONTENT:\n\n{httpx_response.text}"

    else:
        organic_results = get_serpapi_results(query_or_url)

        prompt_header = EXTRACT_PDF_URL_PROMPT
        prompt_context = f"SERPAPI SEARCH RESULTS: {json.dumps(organic_results)}"

    prompt = [
        {
            "content": prompt_header.format(AGENT_ALIAS=ctx.this_agent.alias),
            "role": "system",
        },
        {
            "content": render_conversation(full_conversation),
            "role": "user",
        },
        {
            "content": prompt_context,
            "role": "user",
        },
        {
            "content": "PLEASE ONLY RETURN A URL AND NO OTHER TEXT.\n\nURL:",
            "role": "system",
        },
    ]
    page_url = (await fast_gpt_completion(prompt=prompt).amaterialize_content()).strip()
    assert_valid_url(page_url)

    ctx.respond(
        pdf_finder_agent.quick_call(
            page_url,
            # TODO Oleksandr: make it possible to pass `branch_from` to `quick_call` and `call` directly
            # TODO Oleksandr: `branch_from` should accept either a message promise or a concrete message or a message id
            #  or even a message sequence (but not your own list of messages ?)
            conversation=ConversationTracker(
                forum, branch_from=await ctx.request_messages.aget_concluding_msg_promise()
            ),
            recursion=recursion - 1,
        )
    )


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
