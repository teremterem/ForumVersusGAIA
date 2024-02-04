# pylint: disable=wrong-import-position
"""
Various settings and utility functions for the forum_versus_gaia project.
"""
from functools import partial

from agentforum.ext.llms.openai import openai_chat_completion
from agentforum.forum import Forum
from dotenv import load_dotenv

load_dotenv()

import promptlayer

async_openai_client = promptlayer.openai.AsyncOpenAI()

FAST_GPT = "gpt-3.5-turbo"  # "gpt-3.5-turbo-1106"
# FAST_GPT = "gpt-4-0125-preview"
SLOW_GPT = "gpt-4-0125-preview"
# SLOW_GPT = "gpt-3.5-turbo"  # "gpt-3.5-turbo-1106"

REMOVE_GAIA_LINKS = True

forum = Forum()

# CAPTURED_PROMPTS = []
# CAPTURING_TASKS = []


zero_temperature_completion = partial(
    openai_chat_completion,
    async_openai_client=async_openai_client,
    temperature=0,
)
# def zero_temperature_completion(prompt, **kwargs):
#     """
#     Chat with OpenAI models using zero temperature. Captures the prompt and response for mocking in the future.
#     """
#     result = openai_chat_completion(
#         prompt=prompt,
#         async_openai_client=async_openai_client,
#         temperature=0,
#         **kwargs,
#     )
#
#     async def _capture_prompt():
#         message_dicts = [_message_to_openai_dict(msg) for msg in await amaterialize_message_sequence(prompt)]
#         response_dict = {
#             "content": await result.amaterialize_content(),
#             **(await result.amaterialize_metadata()).as_dict,
#         }
#         CAPTURED_PROMPTS.append(
#             {
#                 "prompt": message_dicts,
#                 "response": response_dict,
#             }
#         )
#
#     CAPTURING_TASKS.append(asyncio.create_task(_capture_prompt()))
#     return result


fast_gpt_completion = partial(
    zero_temperature_completion,
    model=FAST_GPT,
)

slow_gpt_completion = partial(
    zero_temperature_completion,
    model=SLOW_GPT,
)
