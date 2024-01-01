"""
Utilities for the ForumVersusGaia project.
"""
import os
from functools import lru_cache
from typing import Any, Callable
from typing import Iterable
from urllib.parse import urlparse

import html2text
import httpx
from agentforum.models import Message
from serpapi import GoogleSearch

from forum_versus_gaia.forum_versus_gaia_config import REMOVE_GAIA_LINKS


class NotAUrlError(Exception):
    """
    Raised when a string is not a valid URL. the message should only contain the invalid URL and nothing else (because
    it is most likely a message produced by an LLM explaining why the actual url could not be obtained, which was
    returned by the LLM instead of the URL).
    """


def render_conversation(
    conversation: Iterable[Message], alias_renderer: Callable[[Message], str | None] = lambda msg: msg.sender_alias
) -> str:
    """
    Render a conversation as a string. Whenever alias_renderer returns None for a message, that message is skipped.
    """
    turns = []
    for msg in conversation:
        alias = alias_renderer(msg)
        if alias is None:
            continue
        turns.append(f"{alias}: {msg.content.strip()}")
    return "\n\n".join(turns)


def is_valid_url(text: str) -> bool:
    """
    Returns True if the given text is a valid URL.
    """
    try:
        result = urlparse(text)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def assert_valid_url(url: str) -> None:
    """
    Raises an exception if the given URL is not valid.
    """
    if not is_valid_url(url):
        raise NotAUrlError(url)


def get_httpx_client() -> httpx.AsyncClient:
    """
    Returns a httpx client with the settings we want.
    """
    return httpx.AsyncClient(
        follow_redirects=True,
        verify=False,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/58.0.3029.110 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com/",
        },
    )


@lru_cache
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


def convert_html_to_markdown(html: str, baseurl: str = "") -> str:
    """
    Convert HTML to markdown (the best effort).
    """
    h = html2text.HTML2Text(baseurl=baseurl)
    h.ignore_links = False
    return h.handle(html)
