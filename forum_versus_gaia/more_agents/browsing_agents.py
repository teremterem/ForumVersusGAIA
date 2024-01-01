"""Agents that first use Google (SerpAPI) and then "click" around the web pages (follow urls)."""
import io
import json

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


@forum.agent
async def pdf_finder_agent(ctx: InteractionContext, depth: int = 4, retries: int = 4) -> None:
    """
    Much like a search engine but finds and returns from the internet PDFs that satisfy a search query. Useful when
    the information needed to answer a question is more likely to be found in some kind of PDF document rather than
    a webpage. Input should be a search query. (NOTE: {AGENT_ALIAS} already knows that its job is to look for PDFs,
    so you shouldn’t include the word “PDF” in your query.)
    """
    # pylint: disable=too-many-locals
    if depth <= 0 or retries <= 0:
        # TODO Oleksandr: should it be an exception instead ?
        ctx.respond("I couldn't find a PDF document within a reasonable number of hops.")
        return

    completion_method = slow_gpt_completion if retries < 4 else fast_gpt_completion

    full_conversation = await ctx.request_messages.amaterialize_full_history()
    query_or_url = full_conversation[-1].content.strip()

    if is_valid_url(query_or_url):
        async with get_httpx_client() as httpx_client:
            httpx_response = await httpx_client.get(query_or_url)

        if httpx_response.headers["content-type"] == "application/pdf":
            # pdf was found! returning its text

            print("\nOPENING PDF FROM URL: ", query_or_url)

            pdf_reader = pypdf.PdfReader(io.BytesIO(httpx_response.content))
            pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            ctx.respond(pdf_text, success=True)
            return

        if "text/html" not in httpx_response.headers["content-type"]:
            raise RuntimeError(
                f"Expected a PDF or HTML document but got {httpx_response.headers['content-type']} instead."
            )

        print("\nNAVIGATING TO: ", query_or_url)

        prompt_header = EXTRACT_PDF_URL_FROM_PAGE_PROMPT
        prompt_context = (
            f"BELOW IS THE CONTENT OF A WEB PAGE FOUND AT {query_or_url}\n=====\n\n"
            f"{convert_html_to_markdown(httpx_response.text, baseurl=query_or_url)}"
        )

    else:
        print("\nSEARCHING PDF: ", query_or_url)

        organic_results = get_serpapi_results(query_or_url)

        prompt_header = EXTRACT_PDF_URL_PROMPT
        prompt_context = f"SERPAPI SEARCH RESULTS: {json.dumps(organic_results)}"

    prompt = [
        {
            "content": prompt_header.format(AGENT_ALIAS=ctx.this_agent.alias),
            "role": "system",
        },
        {
            "content": "BELOW ARE THE STEPS THAT WERE TRIED SO FAR\n=====\n\n"
            + render_conversation(
                full_conversation,
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
            "content": (
                "PLEASE ONLY RETURN A URL AND NO OTHER TEXT. "
                "MAKE SURE NOT TO RETURN THE URLS THAT WERE ALREADY TRIED.\n\nURL:"
            ),
            "role": "system",
        },
    ]
    page_url = await completion_method(prompt=prompt, pl_tags=[f"d{depth},r{retries}"]).amaterialize_content()
    page_url = page_url.strip()

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
