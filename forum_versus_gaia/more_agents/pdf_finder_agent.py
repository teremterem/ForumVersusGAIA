"""
This module contains an agent that finds PDF documents on the internet.
"""
import io
import json
from typing import Optional

import pypdf
from agentforum.forum import InteractionContext

from forum_versus_gaia.forum_versus_gaia_config import forum, slow_gpt_completion
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

EXTRACT_PDF_URL_FROM_PAGE_PROMPT = """\
Your name is {AGENT_ALIAS}. You will be provided with the content of a web page that was found via web search with a \
given user query. The user is looking for a PDF document. Your job is to extract from this web page a URL that, in \
your opinion, is the most likely to lead to the PDF document the user is looking for.\
"""

PDF_QUERY_COT_PROMPT = """\
Use the following format:

Thought: you should always think out loud before you come up with a search query
Search Query: the query to use to search for the PDF document
Observation: here you speculate how well your search query might perform

NOTE: You are using a special search engine that already knows that you're looking for PDFs, so you shouldn’t \
include "PDF" or "filetype:pdf" or anything like that in your query. Also, don't try to search for any specific \
information that might be contained in the PDF, just search for the PDF itself.

Begin!

Thought:\
"""

# MAX_RETRIES = 3
MAX_DEPTH = 7


@forum.agent
async def pdf_finder_agent(ctx: InteractionContext, beacon: Optional[str] = None) -> None:
    """
    Much like a search engine but finds and returns from the internet PDFs that satisfy a search query. Useful when
    the information needed to answer a question is more likely to be found in some kind of PDF document rather than
    a webpage.
    """
    prompt = [
        {
            "content": (
                f"Your name is {ctx.this_agent.alias} and your job function is to use a search engine to find PDF "
                "documents that are needed to answer the user's question. Here is the question:"
            ),
            "role": "system",
        },
        {
            "content": render_conversation(
                await ctx.request_messages.amaterialize_full_history(), alias_resolver="USER"
            ),
            "role": "user",
        },
        {
            "content": PDF_QUERY_COT_PROMPT,
            "role": "system",
        },
    ]
    query = await slow_gpt_completion(prompt=prompt, stop="\nObservation:", pl_tags=["START"]).amaterialize_content()
    query = query.split("Search Query:")[1]
    query = query.split("\n\n")[0].strip()

    browsing_agent.quick_call(
        query,
        beacon=beacon,
        # TODO Oleksandr: `branch_from` should accept either a message promise or a concrete message or a message
        #  id or even a message sequence (but not your own list of messages ?)
        branch_from=await ctx.request_messages.aget_concluding_msg_promise(),
    )


@forum.agent
async def browsing_agent(ctx: InteractionContext, depth: int = MAX_DEPTH, beacon: Optional[str] = None) -> None:
    # TODO Oleksandr: add a docstring
    from forum_versus_gaia.gaia_agent import gaia_agent  # pylint: disable=import-outside-toplevel,cyclic-import

    if depth <= 0:
        # TODO Oleksandr: should be an exception
        gaia_agent.quick_call(
            "I couldn't find a PDF document within a reasonable number of steps.",
            beacon=beacon,
            failure=True,
            branch_from=await ctx.request_messages.aget_concluding_msg_promise(),
        )
        return

    query_or_url = await ctx.request_messages.amaterialize_concluding_content()
    already_tried_urls = await acollect_tried_urls(ctx)

    if is_valid_url(query_or_url):
        async with get_httpx_client() as httpx_client:
            httpx_response = await httpx_client.get(query_or_url)

        if "application/pdf" in httpx_response.headers["content-type"]:
            # pdf was found! returning its text
            print("\n\033[90mOPENING PDF FROM URL:", query_or_url, "\033[0m")

            pdf_reader = pypdf.PdfReader(io.BytesIO(httpx_response.content))
            pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages])

            is_pdf_correct = await averify_pdf_is_correct(ctx, pdf_text)
            if is_pdf_correct.upper() != "MATCH":
                # TODO Oleksandr: should be an exception (and move inside averify_pdf aka aassert_pdf)
                gaia_agent.quick_call(
                    f"Expected a PDF or HTML document but got {httpx_response.headers['content-type']} instead.",
                    beacon=beacon,
                    failure=True,
                    branch_from=await ctx.request_messages.aget_concluding_msg_promise(),
                )
                return

            gaia_agent.quick_call(
                pdf_text,
                beacon=beacon,
                branch_from=await ctx.request_messages.aget_concluding_msg_promise(),
            )
            return

        if "text/html" not in httpx_response.headers["content-type"]:
            # TODO Oleksandr: should be an exception
            gaia_agent.quick_call(
                f"Expected a PDF or HTML document but got {httpx_response.headers['content-type']} instead.",
                beacon=beacon,
                failure=True,
                branch_from=await ctx.request_messages.aget_concluding_msg_promise(),
            )
            return

        print("\n\033[90mNAVIGATING TO:", query_or_url, "\033[0m")

        prompt_header_template = EXTRACT_PDF_URL_FROM_PAGE_PROMPT
        prompt_context = convert_html_to_markdown(httpx_response.text, baseurl=query_or_url)

    else:
        print("\n\033[90mSEARCHING PDF:", query_or_url, "\033[0m")

        organic_results = get_serpapi_results(query_or_url)
        organic_results = [result for result in organic_results if result["link"].strip() not in already_tried_urls]

        prompt_header_template = EXTRACT_PDF_URL_PROMPT
        prompt_context = f"SERPAPI SEARCH RESULTS: {json.dumps(organic_results)}"

    prompt_context = remove_tried_urls_in_markdown(prompt_context, already_tried_urls)
    page_url = await talk_to_gpt(
        ctx=ctx,
        prompt_header_template=prompt_header_template,
        prompt_context=prompt_context,
        pl_tags=[f"d{depth}"],
    )

    if not is_valid_url(page_url):
        # TODO Oleksandr: should be an exception (use assert_valid_url instead of is_valid_url)
        gaia_agent.quick_call(
            page_url,
            beacon=beacon,
            failure=True,
            branch_from=await ctx.request_messages.aget_concluding_msg_promise(),
        )
        return

    browsing_agent.quick_call(
        page_url,
        depth=depth - 1,
        beacon=beacon,
        branch_from=await ctx.request_messages.aget_concluding_msg_promise(),
    )


async def talk_to_gpt(
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
                alias_resolver=lambda msg: (
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
    page_url = await slow_gpt_completion(prompt=prompt, pl_tags=pl_tags).amaterialize_content()
    return page_url.strip()


async def acollect_tried_urls(ctx: InteractionContext) -> set[str]:
    """
    Collect URLs that were already tried by the agent.
    """
    return {
        msg.content.strip()
        for msg in await ctx.request_messages.amaterialize_full_history()
        if msg.sender_alias == ctx.this_agent.alias and is_valid_url(msg.content.strip())
    }


def remove_tried_urls_in_markdown(prompt_context: str, tried_urls: set[str]) -> str:
    """
    Remove URLs that were already tried from the prompt_context.
    """
    for url in tried_urls:
        prompt_context = prompt_context.replace(f"({url})", "(#)")
    return prompt_context


async def averify_pdf_is_correct(ctx: InteractionContext, pdf_text: str) -> str:
    """
    Ask GPT-4 if this PDF is the one that was referenced in the original question.
    """
    print("\n\033[90mVERIFYING PDF...\033[0m")

    answer = await slow_gpt_completion(
        prompt=[
            {
                "content": (
                    "You are an AI assistant and you are good at validating if PDF documents match what the user is "
                    "asking for. Below is a PDF document."
                ),
                "role": "system",
            },
            {
                "content": pdf_text,
                "role": "user",
            },
            {
                "content": "And here is what the user asked for.",
                "role": "system",
            },
            {
                "content": render_conversation(
                    (await ctx.request_messages.amaterialize_full_history())[:2], alias_resolver="USER"
                ),
                "role": "user",
            },
            {
                "content": (
                    "If the PDF document above matches the user's request then respond with only one word - MATCH, "
                    "and if it is not a match then you are free to respond with any text you want.\n"
                    "\n"
                    "YOUR ANSWER:"
                ),
                "role": "system",
            },
        ],
        pl_tags=["CHECK_PDF"],
    ).amaterialize_content()
    return answer.strip()
