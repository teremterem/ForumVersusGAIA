# pylint: disable=unused-import
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

if __name__ == "__main__":
    from forum_versus_gaia.gaia_agent import amain

    asyncio.run(amain())
