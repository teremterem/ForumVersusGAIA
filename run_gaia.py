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
        "The book with the doi 10.1353/book.24372 concerns a certain neurologist. According to chapter 2 of the "
        "book, what author influenced this neurologist’s belief in “endopsychic myths”? Give the last name only."
    )
    await arun_assistant(question)


if __name__ == "__main__":
    asyncio.run(amain())
