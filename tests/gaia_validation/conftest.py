"""
Pytest configuration with functions to mock calls to OpenAI.
"""

import copy
import importlib
import re
import sys
from functools import partial
from glob import glob
from pathlib import Path
from typing import Any, Optional, Iterable
from unittest.mock import patch, MagicMock

import pytest
from agentforum.ext.llms.openai import _message_to_openai_dict, _OpenAIStreamedMessage
from agentforum.typing import MessageType
from agentforum.utils import amaterialize_message_sequence

from forum_versus_gaia.forum_versus_gaia_config import REMOVE_GAIA_LINKS
from forum_versus_gaia.utils import ContentMismatchError


@pytest.fixture(autouse=True)
def patch_openai() -> None:
    """
    Patch the OpenAI API calls to use captured responses.
    """
    mocking_data = _load_gaia_mocking_data()
    with (
        patch(
            "agentforum.ext.llms.openai._make_openai_request",
            side_effect=partial(_make_openai_request_mock, mocking_data["openai"]),
        ),
        patch(
            "forum_versus_gaia.utils.get_serpapi_results",
            side_effect=partial(_get_serpapi_results_mock, mocking_data["serpapi"]),
        ),
        patch(
            "forum_versus_gaia.utils.adownload_from_web",
            side_effect=partial(_adownload_from_web_mock, mocking_data["web"]),
        ),
    ):
        yield


# noinspection PyProtectedMember,PyUnusedLocal
async def _make_openai_request_mock(
    captured_responses: dict[tuple[tuple[tuple[str, Any], ...], ...], dict[str, Any]],
    prompt: MessageType,
    streamed_message: _OpenAIStreamedMessage,
    async_openai_client: Optional[Any] = None,
    stream: bool = False,
    n: int = 1,
    **kwargs,
) -> None:
    # pylint: disable=protected-access,unused-argument,too-many-arguments
    with _OpenAIStreamedMessage._Producer(streamed_message) as token_producer:
        message_dicts = [_message_to_openai_dict(msg) for msg in await amaterialize_message_sequence(prompt)]
        prompt_key = _convert_prompt_to_captured_key(message_dicts)
        response = captured_responses[prompt_key]
        if stream:
            tokens = re.split(r"(?<=\S)(?=\s+)", response["content"])
            for token in tokens:
                data = MagicMock()
                data.choices[0].delta.content = token
                data.choices[0].delta.role = response["openai_role"]
                data.choices[0].message.content = token
                data.choices[0].message.role = response["openai_role"]
                data.model_dump.return_value = {
                    "choices": [
                        {
                            "delta": {
                                "content": token,
                                "role": response["openai_role"],
                            }
                        }
                    ]
                }
                token_producer.send(data)
        else:
            # send the whole response as a single "token"
            data = MagicMock()
            data.choices[0].delta.content = response["content"]
            data.choices[0].delta.role = response["openai_role"]
            data.choices[0].message.content = response["content"]
            data.choices[0].message.role = response["openai_role"]
            data.model_dump.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": response["content"],
                            "role": response["openai_role"],
                        }
                    }
                ]
            }
            token_producer.send(data)


def _get_serpapi_results_mock(
    captured_responses: dict[tuple[str, bool], list[dict[str, Any]]],
    query: str,
    remove_gaia_links: bool = REMOVE_GAIA_LINKS,
) -> list[dict[str, Any]]:
    return copy.deepcopy(captured_responses[(query, remove_gaia_links)])


async def _adownload_from_web_mock(captured_responses: dict[str, tuple[str, str]], url: str) -> tuple[str, bool]:
    content_type, content = captured_responses[url]
    if "application/pdf" in content_type:
        return content, True

    if "text/html" not in content_type:
        raise ContentMismatchError(
            f"Expected a PDF or HTML document but got {content_type} instead.",
            page_url=url,
        )

    return content, False


def _convert_prompt_to_captured_key(prompt: Iterable[dict[str, Any]]) -> tuple[tuple[tuple[str, Any], ...], ...]:
    """
    Convert the prompt to a key for the captured response dictionary.
    """
    return tuple(tuple(sorted(d.items())) for d in prompt)


def _load_gaia_mocking_data() -> dict[str, Any]:
    mocking_data_dir = Path("../gaia_mocking_data")
    sys.path.append(str(mocking_data_dir))

    mocking_data = {
        "openai": {},
        "serpapi": {},
        "web": {},
    }
    module_files = glob("*.py", root_dir=mocking_data_dir)
    for module_file in module_files:
        module = importlib.import_module(module_file[:-3])
        for openai_response in module.CAPTURED["openai"]:
            prompt_key = _convert_prompt_to_captured_key(openai_response["prompt"])
            mocking_data["openai"][prompt_key] = openai_response["response"]
        for serpapi_response in module.CAPTURED["serpapi"]:
            serpapi_key = (serpapi_response["query"], serpapi_response["remove_gaia_links"])
            mocking_data["serpapi"][serpapi_key] = serpapi_response["organic_results"]
        for web_response in module.CAPTURED["web"]:
            mocking_data["web"][web_response["url"]] = (web_response["content_type"], web_response["content"])

    return mocking_data
