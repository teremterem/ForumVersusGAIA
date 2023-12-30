"""
Utilities for the ForumVersusGaia project.
"""
from typing import Iterable

from agentforum.models import Message


class NotAUrlError(Exception):
    """
    Raised when a string is not a valid URL. the message should only contain the invalid URL and nothing else (because
    it is most likely a message produced by an LLM explaining why the actual url could not be obtained, which was
    returned by the LLM instead of the URL).
    """


def render_conversation(conversation: Iterable[Message]) -> str:
    """Render a conversation as a string."""
    return "\n\n".join([f"{msg.sender_alias}: {msg.content.strip()}" for msg in conversation])
