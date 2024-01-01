"""Agents that first use Google (SerpAPI) and then "click" around the web pages (follow urls)."""
import io
import json
from typing import Callable

import pypdf
from agentforum.forum import InteractionContext

from forum_versus_gaia.forum_versus_gaia_config import forum, slow_gpt_completion, fast_gpt_completion
from forum_versus_gaia.utils import (
    render_conversation,
    get_serpapi_results,
    get_httpx_client,
    is_valid_url,
    convert_html_to_markdown,
)

EXTRACT_PDF_URL_PROMPT = """\
Your name is {AGENT_ALIAS}. You will be provided with a SerpAPI JSON response that contains a list of search results \
for a given user query. The user is looking for a PDF document. Your job is to extract a URL that, in your opinion, \
is the most likely to contain the PDF document the user is looking for.\
"""

# EXTRACT_URL_PROMPT = """\
# Your name is {AGENT_ALIAS}. You will be provided with a SerpAPI JSON response that contains a list of search results \
# for a given user query. The user is looking for an answer to the ORIGINAL QUESTION (you will se it below). Your job \
# is to extract a URL that, in your opinion, is the most likely to contain the answer the user is looking for.\
# """

EXTRACT_PDF_URL_FROM_PAGE_PROMPT = """\
Your name is {AGENT_ALIAS}. You will be provided with the content of a web page that was found via web search with a \
given user query. The user is looking for a PDF document. Your job is to extract from this web page a URL that, in \
your opinion, is the most likely to lead to the PDF document the user is looking for.\
"""

MAX_DEPTH = 5
MAX_RETRIES = 3


@forum.agent
async def pdf_finder_agent(ctx: InteractionContext, depth: int = MAX_DEPTH, retries: int = MAX_RETRIES) -> None:
    """
    Much like a search engine but finds and returns from the internet PDFs that satisfy a search query. Useful when
    the information needed to answer a question is more likely to be found in some kind of PDF document rather than
    a webpage. Input should be a search query. (NOTE: {AGENT_ALIAS} already knows that its job is to look for PDFs,
    so you shouldn’t include "PDF" or "filetype:pdf" or anything like that in your query.)
    """
    # pylint: disable=too-many-locals
    if depth <= 0 or retries <= 0:
        # TODO Oleksandr: should it be an exception instead ?
        ctx.respond("I couldn't find a PDF document within a reasonable number of steps.")
        return

    is_a_retry = retries < MAX_RETRIES
    completion_method = slow_gpt_completion if is_a_retry else fast_gpt_completion

    query_or_url = await ctx.request_messages.amaterialize_concluding_content()

    if is_valid_url(query_or_url):
        async with get_httpx_client() as httpx_client:
            httpx_response = await httpx_client.get(query_or_url)

        if httpx_response.headers["content-type"] == "application/pdf":
            # pdf was found! returning its text

            print("\n\033[90mOPENING PDF FROM URL:", query_or_url, "\033[0m")

            pdf_reader = pypdf.PdfReader(io.BytesIO(httpx_response.content))
            pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            ctx.respond(pdf_text, success=True)
            return

        if "text/html" not in httpx_response.headers["content-type"]:
            raise RuntimeError(
                f"Expected a PDF or HTML document but got {httpx_response.headers['content-type']} instead."
            )

        print("\n\033[90mNAVIGATING TO:", query_or_url, "\033[0m")

        prompt_header_template = EXTRACT_PDF_URL_FROM_PAGE_PROMPT
        prompt_context = convert_html_to_markdown(httpx_response.text, baseurl=query_or_url)

    else:
        print("\n\033[90mSEARCHING PDF:", query_or_url, "\033[0m")

        organic_results = get_serpapi_results(query_or_url)

        prompt_header_template = EXTRACT_PDF_URL_PROMPT
        prompt_context = f"SERPAPI SEARCH RESULTS: {json.dumps(organic_results)}"

    page_url = await talk_to_gpt(
        completion_method=completion_method,
        ctx=ctx,
        prompt_header_template=prompt_header_template,
        prompt_context=prompt_context,
        pl_tags=[f"d{depth},r{retries}"],
    )

    if is_valid_url(page_url):
        recursive_resp_promise = await pdf_finder_agent.quick_call(
            page_url,
            branch_from=await ctx.request_messages.aget_concluding_msg_promise(),
            depth=depth - 1,
            retries=retries,
        ).aget_concluding_msg_promise()

        # TODO Oleksandr: avoid having to specify that it is `.metadata` - it's confusing,
        #  you don't specify it when you set it (OR make sure to specify `.metadata` everywhere)
        if not getattr((await recursive_resp_promise.amaterialize()).metadata, "success", False):
            # PDF was not found, let's try again
            current_request = await ctx.request_messages.amaterialize_concluding_message()
            recursive_resp_promise = await pdf_finder_agent.quick_call(
                current_request,
                override_sender_alias=current_request.get_original_msg().sender_alias,
                branch_from=recursive_resp_promise,
                depth=depth,
                retries=retries - 1,
            ).amaterialize_concluding_message()

        ctx.respond(recursive_resp_promise)

    else:
        # it's not a url, it's most likely a message that tells that something went wrong
        ctx.respond(page_url)


async def talk_to_gpt(
    completion_method: Callable,
    ctx: InteractionContext,
    prompt_header_template: str,
    prompt_context: str,
    pl_tags: list[str] = (),
):
    """
    Talk to GPT to get the next URL to navigate to.
    """
    prompt = [
        {
            "content": prompt_header_template.format(AGENT_ALIAS=ctx.this_agent.alias),
            "role": "system",
        },
        {
            "content": render_conversation(
                await ctx.request_messages.amaterialize_full_history(),
                alias_renderer=lambda msg: (
                    f"{msg.sender_alias} - {'NAVIGATE TO' if is_valid_url(msg.content.strip()) else 'RESULT'}"
                    if msg.sender_alias == ctx.this_agent.alias
                    else "USER"
                ),
            ),
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
    page_url = await completion_method(prompt=prompt, pl_tags=pl_tags).amaterialize_content()
    return page_url.strip()
