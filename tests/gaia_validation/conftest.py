"""
Pytest configuration with functions to mock calls to OpenAI.
"""

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


@pytest.fixture  # (autouse=True)
def patch_openai() -> None:
    """
    Patch the OpenAI API calls to use captured responses.
    """
    captured_responses = _load_captured_openai_responses()
    with patch(
        "agentforum.ext.llms.openai._make_openai_request",
        side_effect=partial(_make_patched_openai_request, captured_responses),
    ):
        yield


# noinspection PyProtectedMember,PyUnusedLocal
async def _make_patched_openai_request(
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


def _convert_prompt_to_captured_key(prompt: Iterable[dict[str, Any]]) -> tuple[tuple[tuple[str, Any], ...], ...]:
    """
    Convert the prompt to a key for the captured response dictionary.
    """
    return tuple(tuple(sorted(d.items())) for d in prompt)


def _load_captured_openai_responses() -> dict[tuple[tuple[tuple[str, Any], ...], ...], dict[str, Any]]:
    """
    Load the captured OpenAI responses from the captured_prompts directory.
    """
    captured_responses_dir = Path("../gaia_mocking_data")
    sys.path.append(str(captured_responses_dir))

    captured_responses = {}
    module_files = glob("*.py", root_dir=captured_responses_dir)
    for module_file in module_files:
        module = importlib.import_module(module_file[:-3])
        for prompt_response in module.CAPTURED["openai"]:
            prompt_key = _convert_prompt_to_captured_key(prompt_response["prompt"])
            captured_responses[prompt_key] = prompt_response["response"]

    return captured_responses
