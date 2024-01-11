# pylint: disable=import-outside-toplevel,unused-import
"""Run GAIA solver."""
import asyncio

# noinspection PyUnresolvedReferences
import readline
import warnings

# ATTENTION! This import must go before any other imports from this project
# noinspection PyUnresolvedReferences
from forum_versus_gaia import forum_versus_gaia_config

# TODO Oleksandr: get rid of this warning suppression when PromptLayer doesn't produce "Expected Choice but got dict"
#  warning anymore
warnings.filterwarnings("ignore", module="pydantic")


async def amain() -> None:
    """
    Run the assistant on a question from the GAIA dataset.
    """
    from forum_versus_gaia.gaia_agent import arun_assistant

    question = (
        "In Valentina Re’s contribution to the 2017 book “World Building: Transmedia, Fans, Industries”, what "
        "horror movie does the author cite as having popularized metalepsis between a dream world and reality? "
        "Use the complete name with article if any."
    )
    await arun_assistant(question)


if __name__ == "__main__":
    asyncio.run(amain())
