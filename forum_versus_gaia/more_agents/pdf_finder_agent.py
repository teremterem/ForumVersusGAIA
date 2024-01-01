"""
This module contains an agent that finds PDF documents on the internet.
"""
import io
import json

import pypdf
from agentforum.forum import InteractionContext

from forum_versus_gaia.forum_versus_gaia_config import forum, fast_gpt_completion, slow_gpt_completion
from forum_versus_gaia.utils import (
    render_conversation,
    get_serpapi_results,
    get_httpx_client,
    is_valid_url,
    convert_html_to_markdown,
    TooManyStepsError,
    ForumVersusGaiaError,
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

# TODO Oleksandr: simplify this prompt - there is only one "tool", no need for it to be so fancy
CHOOSE_TOOL_PROMPT = """\
Answer the following questions as best you can. You have access to the following tools:

{AGENT_DESCRIPTIONS}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{AGENT_NAMES}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!\
"""

MAX_DEPTH = 5
MAX_RETRIES = 3


@forum.agent
async def pdf_finder_agent(ctx: InteractionContext) -> None:
    """
    Much like a search engine but finds and returns from the internet PDFs that satisfy a search query. Useful when
    the information needed to answer a question is more likely to be found in some kind of PDF document rather than
    a webpage. Input should be a search query. (NOTE: {AGENT_ALIAS} already knows that its job is to look for PDFs,
    so you shouldn’t include "PDF" or "filetype:pdf" or anything like that in your query. Also, don't try to search
    for any specific information that might be contained in the PDF, just search for the PDF itself.)
    """
    agent_names = ctx.this_agent.alias
    agent_descriptions = f"{ctx.this_agent.alias}: {ctx.this_agent.description}"

    # TODO Oleksandr: find a prompt format that allows full chat history to be passed ?
    question = await ctx.request_messages.amaterialize_concluding_content()
    question = question.strip()

    prompt = [
        {
            "content": CHOOSE_TOOL_PROMPT.format(AGENT_NAMES=agent_names, AGENT_DESCRIPTIONS=agent_descriptions),
            "role": "system",
        },
        {
            "content": f"Question: {question}\nThought:",
            "role": "user",
        },
    ]
    query_msg_content = await fast_gpt_completion(
        prompt=prompt, stop="\nObservation:", pl_tags=["START"]
    ).amaterialize_content()
    query = query_msg_content.split("Action Input:")[1].strip()

    try:
        pdf_msg = await pdf_finder_no_proxy.quick_call(
            query,
            # TODO Oleksandr: `branch_from` should accept either a message promise or a concrete message or a message
            #  id or even a message sequence (but not your own list of messages ?)
            branch_from=await ctx.request_messages.aget_concluding_msg_promise(),
            first_try=True,
            retries=1,
        ).amaterialize_concluding_message()

    except ForumVersusGaiaError:
        query_msg_content = await slow_gpt_completion(
            prompt=prompt, stop="\nObservation:", pl_tags=["RETRY"]
        ).amaterialize_content()
        query = query_msg_content.split("Action Input:")[1].strip()

        pdf_msg = await pdf_finder_no_proxy.quick_call(
            query,
            # TODO Oleksandr: `branch_from` should accept either a message promise or a concrete message or a message
            #  id or even a message sequence (but not your own list of messages ?)
            branch_from=await ctx.request_messages.aget_concluding_msg_promise(),
            first_try=False,
        ).amaterialize_concluding_message()

    ctx.respond(pdf_msg)


@forum.agent
async def pdf_finder_no_proxy(
    ctx: InteractionContext, first_try: bool, depth: int = MAX_DEPTH, retries: int = MAX_RETRIES
) -> None:
    """
    TODO Oleksandr: introduce the concept of proxy agents at the level of the AgentForum framework
    """
    # pylint: disable=too-many-locals
    if depth <= 0 or retries <= 0:
        error_message = "I couldn't find a PDF document within a reasonable number of steps."
        if first_try:
            raise TooManyStepsError(error_message)
        ctx.respond(error_message)
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
        # query_or_url = query_or_url.replace('"', "")
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
        first_try=first_try,
        pl_tags=[f"d{depth},r{retries}"],
    )

    if is_valid_url(page_url):
        recursive_resp_promise = await pdf_finder_no_proxy.quick_call(
            page_url,
            branch_from=await ctx.request_messages.aget_concluding_msg_promise(),
            first_try=first_try,
            depth=depth - 1,
            retries=retries,
        ).aget_concluding_msg_promise()

        # TODO Oleksandr: avoid having to specify that it is `.metadata` - it's confusing,
        #  you don't specify it when you set it (OR make sure to specify `.metadata` everywhere)
        if not getattr((await recursive_resp_promise.amaterialize()).metadata, "success", False):
            # PDF was not found, let's try again
            current_request = await ctx.request_messages.amaterialize_concluding_message()
            recursive_resp_promise = await pdf_finder_no_proxy.quick_call(
                current_request,
                override_sender_alias=current_request.get_original_msg().sender_alias,
                branch_from=recursive_resp_promise,
                first_try=first_try,
                depth=depth,
                retries=retries - 1,
            ).amaterialize_concluding_message()

        ctx.respond(recursive_resp_promise)

    else:
        # it's not a url, it's most likely a message that tells that something went wrong
        if first_try:
            raise ForumVersusGaiaError(page_url)
        ctx.respond(page_url)


async def talk_to_gpt(
    ctx: InteractionContext,
    prompt_header_template: str,
    prompt_context: str,
    first_try: bool,
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
            "content": (
                "PLEASE ONLY RETURN A URL AND NO OTHER TEXT. "
                "MAKE SURE NOT TO RETURN THE URLS THAT WERE ALREADY TRIED.\n\nURL:"
            ),
            "role": "system",
        },
    ]
    completion_method = fast_gpt_completion if first_try else slow_gpt_completion
    page_url = await completion_method(prompt=prompt, pl_tags=pl_tags).amaterialize_content()
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
