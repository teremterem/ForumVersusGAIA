"""
This module contains an agent that finds PDF documents on the internet.
"""
import asyncio
import io
import json
import secrets
from typing import Optional

import pypdf
from agentforum.forum import InteractionContext, USER_ALIAS

from forum_versus_gaia.forum_versus_gaia_config import forum, slow_gpt_completion
from forum_versus_gaia.utils import (
    render_conversation,
    get_serpapi_results,
    get_httpx_client,
    is_valid_url,
    convert_html_to_markdown,
)

MAX_RETRIES = 3
MAX_DEPTH = 7

RESPONSES: dict[str, asyncio.Queue] = {}


@forum.agent
async def pdf_finder_agent(ctx: InteractionContext, beacon: Optional[str] = None, failure: bool = False) -> None:
    """
    Much like a search engine but finds and returns from the internet PDFs that satisfy a search query. Useful when
    the information needed to answer a question is more likely to be found in some kind of PDF document rather than
    a webpage.
    """
    if beacon is not None:
        RESPONSES.pop(beacon).put_nowait((ctx.request_messages, failure))
        return

    prompt = [
        {
            "content": (
                f"Your name is {ctx.this_agent.alias} and your job function is to use a search engine to find PDF "
                "documents that are needed to answer the user's question. Here is the question:"
            ),
            "role": "system",
        },
        {
            "content": render_conversation(await ctx.request_messages.amaterialize_as_list()),
            "role": "user",
        },
        {
            "content": (
                "Use the following format:\n"
                "\n"
                "Thought: you should always think out loud before you come up with a search query\n"
                "Search Query: the query to use to search for the PDF document\n"
                "\n"
                'If you need to search for multiple PDF documents then just repeat "Search Query:" multiple times.\n'
                "\n"
                "NOTE: You are using a special search engine that already knows that you're looking for PDFs, so "
                'you shouldn\'t include "PDF" or "filetype:pdf" or anything like that in your query. Also, don\'t '
                "try to search for any specific information that might be contained in the PDF, just search for the "
                "PDF itself.\n"
                "\n"
                "Begin!\n"
                "\n"
                "Thought:"
            ),
            "role": "system",
        },
    ]
    queries = await slow_gpt_completion(prompt=prompt, pl_tags=["START"]).amaterialize_content()

    for query in queries.split("Search Query:")[1:]:
        query = query.split("\n\n")[0].strip()

        responses = ctx.request_messages  # this is just for convenience - it will be overwritten later
        for _ in range(MAX_RETRIES):
            beacon = secrets.token_hex(4)
            RESPONSES[beacon] = asyncio.Queue()

            pdf_browsing_agent.quick_call(
                query,
                beacon=beacon,
                # TODO Oleksandr: `branch_from` should accept either a message promise or a concrete message or a
                #  message id or even a message sequence (but not your own list of messages ?)
                branch_from=await responses.aget_concluding_msg_promise(),
            )

            responses, failure = await RESPONSES[beacon].get()
            if not failure:
                break

        ctx.respond(responses)


@forum.agent(alias="BROWSING_AGENT")
async def pdf_browsing_agent(ctx: InteractionContext, depth: int = MAX_DEPTH, beacon: Optional[str] = None) -> None:
    """
    Navigates the web to find a PDF document that satisfies the user's request.
    """
    if depth <= 0:
        # TODO Oleksandr: should be an exception
        pdf_finder_agent.quick_call(
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
            print("\n\033[90m📗 READING PDF FROM:", query_or_url, "\033[0m")

            pdf_reader = pypdf.PdfReader(io.BytesIO(httpx_response.content))
            pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages])

            pdf_snippets = await aextract_pdf_snippets(ctx, pdf_text)
            if pdf_snippets.upper() == "MISMATCH":
                # TODO Oleksandr: should be an exception (and move inside aextract_pdf_snippets)
                pdf_finder_agent.quick_call(
                    "This PDF document does not contain any relevant information.",
                    beacon=beacon,
                    failure=True,
                    branch_from=await ctx.request_messages.aget_concluding_msg_promise(),
                )
                return

            pdf_finder_agent.quick_call(
                pdf_snippets,
                beacon=beacon,
                branch_from=await ctx.request_messages.aget_concluding_msg_promise(),
            )
            return

        if "text/html" not in httpx_response.headers["content-type"]:
            # TODO Oleksandr: should be an exception
            pdf_finder_agent.quick_call(
                f"Expected a PDF or HTML document but got {httpx_response.headers['content-type']} instead.",
                beacon=beacon,
                failure=True,
                branch_from=await ctx.request_messages.aget_concluding_msg_promise(),
            )
            return

        print("\n\033[90m🔗 NAVIGATING TO:", query_or_url, "\033[0m")

        prompt_header_template = (
            "Your name is {AGENT_ALIAS}. You will be provided with the content of a web page that was found via "
            "web search with a given user query. The user is looking for a PDF document. Your job is to extract "
            "from this web page a URL that, in your opinion, is the most likely to lead to the PDF document the "
            "user is looking for."
        )
        prompt_context = convert_html_to_markdown(httpx_response.text, baseurl=query_or_url)
        prompt_context = remove_tried_urls_in_markdown(prompt_context, already_tried_urls)

    else:
        print("\n\033[90m🔍 LOOKING FOR PDF:", query_or_url, "\033[0m")

        organic_results = get_serpapi_results(query_or_url)
        organic_results = [result for result in organic_results if result["link"].strip() not in already_tried_urls]

        prompt_header_template = (
            "Your name is {AGENT_ALIAS}. You will be provided with a SerpAPI JSON response that contains a list "
            "of search results for a given user query. The user is looking for a PDF document. Your job is to "
            "extract a URL that, in your opinion, is the most likely to contain the PDF document the user is "
            "looking for."
        )
        prompt_context = json.dumps(organic_results)

    page_url = await ask_gpt_for_url(
        ctx=ctx,
        prompt_header_template=prompt_header_template,
        prompt_context=prompt_context,
        pl_tags=[f"d{depth}"],
    )

    if not is_valid_url(page_url):
        # TODO Oleksandr: should be an exception (use assert_valid_url instead of is_valid_url)
        pdf_finder_agent.quick_call(
            page_url,
            beacon=beacon,
            failure=True,
            branch_from=await ctx.request_messages.aget_concluding_msg_promise(),
        )
        return

    pdf_browsing_agent.quick_call(
        page_url,
        depth=depth - 1,
        beacon=beacon,
        branch_from=await ctx.request_messages.aget_concluding_msg_promise(),
    )


async def ask_gpt_for_url(
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
            "content": await render_user_utterances(ctx),
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


async def aextract_pdf_snippets(ctx: InteractionContext, pdf_text: str) -> str:
    """
    Extract snippets from a PDF document that are relevant to the user's request.
    """
    answer = await slow_gpt_completion(
        prompt=[
            {
                "content": (
                    "You are an AI assistant and you are good at extracting relevant information from PDF documents. "
                    "Below is a PDF document."
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
                "content": await render_user_utterances(ctx),
                "role": "user",
            },
            {
                "content": (
                    "Please extract a snippet or snippets from the PDF document that you think are relevant to the "
                    "user's request. Use the following format:\n"
                    "\n"
                    "PDF TITLE: the title of the pdf\n"
                    "DESCRIPTION: briefly explain what this pdf is about\n"
                    "RELEVANT SNIPPET(S): a snippet or snippets relevant to the user's request (make sure to capture "
                    "a couple of surrounding sentences too)\n"
                    "\n"
                    "If the PDF document does not contain any relevant information then respond with only one word - "
                    "MISMATCH\n"
                    "\n"
                    'ATTENTION! DO NOT ANSWER WITH "MISMATCH" IF YOU WERE ABLE TO FIND EVEN A TINY BIT OF RELEVANT '
                    'INFORMATION. "MISMATCH" is ONLY for cases when there was NOT EVEN A SINGLE PIECE of relevant '
                    "information (ABSOLUTE ZERO)!\n"
                    "\n"
                    "Begin!"
                ),
                "role": "system",
            },
        ],
        pl_tags=["READ_PDF"],
    ).amaterialize_content()
    return answer.strip()


async def render_user_utterances(ctx: InteractionContext) -> str:
    """
    Render user utterances as a string.
    """
    return render_conversation(
        await ctx.request_messages.amaterialize_full_history(),
        alias_resolver=lambda msg: msg.sender_alias if msg.sender_alias == USER_ALIAS else None,
    )
