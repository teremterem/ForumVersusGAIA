"""
Utilities for the ForumVersusGaia project.
"""

import io
import math
import os
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

import html2text
import httpx
import numpy as np
import pypdf
from agentforum.errors import FormattedForumError
from agentforum.models import Freeform
from serpapi import GoogleSearch

from forum_versus_gaia import forum_versus_gaia_config
from forum_versus_gaia.forum_versus_gaia_config import REMOVE_GAIA_LINKS


class ForumVersusGaiaError(FormattedForumError):
    """
    Base class for all exceptions in the ForumVersusGaia project.
    """


class NotAUrlError(ForumVersusGaiaError):
    """
    Raised when a string is not a valid URL. the message should only contain the invalid URL and nothing else (because
    it is most likely a message produced by an LLM explaining why the actual url could not be obtained, which was
    returned by the LLM instead of the URL).
    """


class ContentMismatchError(ForumVersusGaiaError):
    """
    Raised when the content that was found on the web (a PDF, a webpage, etc.) does not match what the user was looking
    for.
    """


class ContentNotFoundError(ForumVersusGaiaError):
    """
    Raised when it was not possible to find the content that the user's request was about.
    """


class ContentAlreadySeenError(ForumVersusGaiaError):
    """
    Raised when the content was already seen by the agent before.
    """


class TooManyStepsError(ForumVersusGaiaError):
    """
    Raised when an agent takes too many steps to complete.
    """


def is_valid_url(text: str) -> bool:
    """
    Returns True if the given text is a valid URL.
    """
    try:
        result = urlparse(text)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def assert_valid_url(url: str, error_class: type[BaseException] = NotAUrlError) -> None:
    """
    Raises an exception if the given URL is not valid.
    """
    if not is_valid_url(url):
        raise error_class(url)


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
    # TODO Oleksandr: make this function async by replacing SerpAPI python client with plain aiohttp ?
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
    if forum_versus_gaia_config.CAPTURE_MOCKING_DATA:
        forum_versus_gaia_config.CAPTURED_DATA["serpapi"].append(
            {
                "query": query,
                "remove_gaia_links": remove_gaia_links,
                "organic_results": organic_results,
            }
        )
    return organic_results


async def adownload_from_web(url: str) -> tuple[str, bool]:
    """
    Download content from the web and return it as a string. If the content is a PDF, return the text extracted from the
    PDF as well. Returns a tuple of the content and a boolean indicating whether the content is a PDF.
    """
    async with get_httpx_client() as httpx_client:
        httpx_response = await httpx_client.get(url)

    if "application/pdf" in httpx_response.headers["content-type"]:
        pdf_reader = pypdf.PdfReader(io.BytesIO(httpx_response.content))
        pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
        if forum_versus_gaia_config.CAPTURE_MOCKING_DATA:
            forum_versus_gaia_config.CAPTURED_DATA["web"].append(
                {
                    "url": url,
                    "content_type": httpx_response.headers["content-type"],
                    "content": pdf_text,
                }
            )
        return pdf_text, True

    if "text/html" not in httpx_response.headers["content-type"]:
        raise ContentMismatchError(
            f"Expected a PDF or HTML document but got {httpx_response.headers['content-type']} instead.",
            page_url=url,
        )

    if forum_versus_gaia_config.CAPTURE_MOCKING_DATA:
        forum_versus_gaia_config.CAPTURED_DATA["web"].append(
            {
                "url": url,
                "content_type": httpx_response.headers["content-type"],
                "content": httpx_response.text,
            }
        )
    return httpx_response.text, False


def convert_html_to_markdown(html: str, baseurl: str = "") -> str:
    """
    Convert HTML to markdown (the best effort).
    """
    h = html2text.HTML2Text(baseurl=baseurl, bodywidth=0)
    h.ignore_links = False
    return h.handle(html)


def calculate_perplexity(openai_metadata: Freeform) -> float:
    """
    Calculate perplexity from the log probabilities of tokens found in a message metadata.
    """
    log_probs = [logprob.logprob for logprob in openai_metadata.openai_logprobs]
    average_log_prob = np.mean(log_probs)
    perplexity = np.exp(-average_log_prob)
    return perplexity


def calculate_geometric_mean_of_probabilities(openai_metadata: Freeform) -> float:
    """
    Calculate the geometric mean of probabilities from the log probabilities of tokens found in a message metadata.
    """
    log_probs = [logprob.logprob for logprob in openai_metadata.openai_logprobs]
    probs = np.exp(log_probs)
    geometric_mean = np.prod(probs) ** (1 / len(probs))
    return geometric_mean


def find_min_probability(openai_metadata: Freeform) -> float:
    """
    Find the minimum probability of tokens found in a message metadata.
    """
    log_probs = [logprob.logprob for logprob in openai_metadata.openai_logprobs]
    min_log_prob = min(log_probs)
    min_probability = math.exp(min_log_prob)
    return min_probability
