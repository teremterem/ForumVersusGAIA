"""
This module contains an agent that finds PDF documents on the internet.
"""

import io
import json

import pypdf
from agentforum.ext.llms.openai import anum_tokens_from_messages
from agentforum.forum import InteractionContext, USER_ALIAS
from agentforum.models import Message
from agentforum.utils import arender_conversation

from forum_versus_gaia.forum_versus_gaia_config import forum, slow_gpt_completion
from forum_versus_gaia.utils import (
    get_serpapi_results,
    get_httpx_client,
    is_valid_url,
    convert_html_to_markdown,
    ContentMismatchError,
    assert_valid_url,
    ContentNotFoundError,
    ForumVersusGaiaError,
)

MAX_RETRIES = 3
MAX_DEPTH = 7

PDF_MAX_TOKENS = 100000
PDF_CHAR_WINDOW = 10000
PDF_CHAR_OVERLAP = 1000


@forum.agent
async def pdf_finder_agent(ctx: InteractionContext) -> None:
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
            "content": await arender_conversation(ctx.request_messages),
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
                "NOTE #1: You are using a special search engine that already knows that you're looking for PDFs, so "
                'you shouldn\'t include "PDF" or "filetype:pdf" or anything like that in your query.\n'
                "NOTE #2: Do not try to search for any specific information that might be contained in the PDF, "
                "just search for the PDF itself.\n"
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
            responses = pdf_browsing_agent.ask(
                query,
                # TODO Oleksandr: `branch_from` should accept either a message promise or a concrete message or a
                #  message id or even a message sequence (but not your own list of messages ?)
                branch_from=await responses.aget_concluding_msg_promise(),
            )
            if not await responses.acontains_errors():
                break

        ctx.respond(responses)


@forum.agent(alias="BROWSING_AGENT")
async def pdf_browsing_agent(ctx: InteractionContext, depth: int = MAX_DEPTH) -> None:
    """
    Navigates the web to find a PDF document that satisfies the user's request.
    """
    if depth <= 0:
        raise ContentNotFoundError("I couldn't find a PDF document within a reasonable number of steps.")

    query_or_url = await ctx.request_messages.amaterialize_concluding_content()
    already_tried_urls = await acollect_tried_urls(ctx)

    if is_valid_url(query_or_url):
        async with get_httpx_client() as httpx_client:
            httpx_response = await httpx_client.get(query_or_url)

        if "application/pdf" in httpx_response.headers["content-type"]:
            # pdf was found! returning its text
            # TODO Oleksandr: introduce the concept of user_proxy_agent to send all these service messages
            #  to that agent instead of just printing them directly to the console
            print(f"\n\033[90m📗 READING PDF FROM: {query_or_url}", end="", flush=True)

            pdf_reader = pypdf.PdfReader(io.BytesIO(httpx_response.content))
            pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages])

            pdf_snippets = await aextract_pdf_snippets(
                pdf_text=pdf_text, user_request=await render_user_utterances(ctx)
            )
            ctx.respond(pdf_snippets, pdf=pdf_text)
            return

        if "text/html" not in httpx_response.headers["content-type"]:
            raise ContentMismatchError(
                f"Expected a PDF or HTML document but got {httpx_response.headers['content-type']} instead."
            )

        print(f"\n\033[90m🔗 NAVIGATING TO: {query_or_url}\033[0m")

        prompt_header_template = (
            "Your name is {AGENT_ALIAS}. You will be provided with the content of a web page that was found via "
            "web search with a given user query. The user is looking for a PDF document. Your job is to extract "
            "from this web page a URL that, in your opinion, is the most likely to lead to the PDF document the "
            "user is looking for."
        )
        prompt_context = convert_html_to_markdown(httpx_response.text, baseurl=query_or_url)
        prompt_context = remove_tried_urls_in_markdown(prompt_context, already_tried_urls)

    else:
        print(f"\n\033[90m🔍 LOOKING FOR PDF: {query_or_url}\033[0m")

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

    assert_valid_url(page_url, error_class=ContentNotFoundError)
    pdf_browsing_agent.tell(
        Message(
            content_template="{page_url}",
            page_url=page_url,
        ),
        depth=depth - 1,
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
        if msg.original_sender_alias == ctx.this_agent.alias and is_valid_url(msg.content.strip())
    }


def remove_tried_urls_in_markdown(prompt_context: str, tried_urls: set[str]) -> str:
    """
    Remove URLs that were already tried from the prompt_context.
    """
    for url in tried_urls:
        prompt_context = prompt_context.replace(f"({url})", "(#)")
    return prompt_context


ALREADY_CHECKED_PDFS: set[str] = set()  # TODO TODO TODO Oleksandr: this is a super temporary solution !!!


async def aextract_pdf_snippets(pdf_text: str, user_request: str) -> str:
    """
    Extract snippets from a PDF document that are relevant to the user's request. If pdf_text is a wrong PDF
    document or does not contain any useful information then ContentMismatchError is raised.
    """
    if pdf_text in ALREADY_CHECKED_PDFS:
        print(" - ALREADY SEEN\033[0m")
        # just return this as if it's a "relevant snippet", as we probably already have relevant snippet(s)
        return "-"
    ALREADY_CHECKED_PDFS.add(pdf_text)

    pdf_msgs = [
        {
            "content": (
                f"=============== PDF START ===============\n"
                f"{pdf_text}\n"
                f"================ PDF END ================"
            ),
            "role": "user",
        },
    ]
    pdf_token_num = await anum_tokens_from_messages(pdf_msgs)
    print(f" - {pdf_token_num} tokens\033[0m")

    if pdf_token_num > PDF_MAX_TOKENS:
        return await apartition_pdf_and_extract_snippets(pdf_text=pdf_text, user_request=user_request)

    answer = await slow_gpt_completion(
        prompt=[
            {
                "content": (
                    "You are an AI assistant and you are good at extracting relevant information from PDF documents. "
                    "Below is a PDF document."
                ),
                "role": "system",
            },
            *pdf_msgs,
            {
                "content": "And here is what the user asked for.",
                "role": "system",
            },
            {
                "content": user_request,
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
    answer = answer.strip()
    if answer.upper() == "MISMATCH":
        raise ContentMismatchError("This PDF document does not contain any relevant information.", pdf=pdf_text)
    return answer


async def apartition_pdf_and_extract_snippets(pdf_text: str, user_request: str) -> str:
    """
    Partition a PDF document into chunks and extract snippets from each chunk that are relevant to the user's request.
    If pdf_text is a wrong PDF document or does not contain any useful information then ContentMismatchError is
    raised.
    """
    raise ForumVersusGaiaError("PDF partitioning is not implemented yet", pdf=pdf_text)
    # pylint: disable=unreachable
    pdf_metadata = await agenerate_metadata_from_pdf_parts(pdf_text=pdf_text, user_request=user_request)
    # TODO TODO TODO Oleksandr
    return pdf_metadata


async def agenerate_metadata_from_pdf_parts(pdf_text: str, user_request: str) -> str:
    """
    Generate metadata for a PDF document from its parts (beginning, middle and end). If pdf_text is a wrong PDF
    document or does not contain any useful information then ContentMismatchError is raised.
    """
    pdf_beginning = pdf_text[:PDF_CHAR_WINDOW]
    pdf_middle = pdf_text[len(pdf_text) // 2 - PDF_CHAR_WINDOW // 2 : len(pdf_text) // 2 + PDF_CHAR_WINDOW // 2]
    pdf_end = pdf_text[-PDF_CHAR_WINDOW:]

    answer = await slow_gpt_completion(
        prompt=[
            {
                "content": (
                    "You are an AI assistant and you are good at explaining what PDF documents are about. "
                    "Below is a PDF document (some parts of it were omitted for brevity)."
                ),
                "role": "system",
            },
            {
                "content": f"{pdf_beginning}...\n\n...{pdf_middle}...\n\n...{pdf_end}",
                "role": "user",
            },
            {
                "content": "And here is what the user asked for.",
                "role": "system",
            },
            {
                "content": user_request,
                "role": "user",
            },
            {
                "content": (
                    "Use the following format to describe the PDF:\n"
                    "\n"
                    "PDF TITLE: the title of the pdf\n"
                    "DESCRIPTION: briefly explain what this pdf is about\n"
                    "\n"
                    "If the PDF document does not seem to be the one that could be used to answer the user's "
                    "question then end your answer with the word MISMATCH\n"
                    "NOTE: You shouldn't judge whether the PDF document contains any relevant information or not "
                    "solely by the presence/absence of the direct answer to the user's question in the content you "
                    "see in the prompt above, because you are not given the full PDF document, you are given only "
                    "parts of it. You should judge based on whether the PDF document in general seems to be the "
                    "relevant one to the user's question or not.\n"
                    "\n"
                    "Begin!"
                ),
                "role": "system",
            },
        ],
        pl_tags=["PDF_METADATA_FROM_PARTS"],
    ).amaterialize_content()
    answer = answer.strip()
    if answer.endswith("\nMISMATCH"):
        raise ContentMismatchError("This PDF document does not seem to be relevant.")
    return answer


async def render_user_utterances(ctx: InteractionContext) -> str:
    """
    Render user utterances as a string.
    """
    return await arender_conversation(
        # TODO TODO TODO Oleksandr: introduce a method that returns full history as an AsyncMessageSequence
        await ctx.request_messages.aget_full_history(),
        alias_resolver=lambda msg: msg.original_sender_alias if msg.original_sender_alias == USER_ALIAS else None,
    )
