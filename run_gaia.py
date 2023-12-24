# pylint: disable=wrong-import-position
"""Run GAIA resolver."""
import asyncio

# noinspection PyUnresolvedReferences
import readline  # pylint: disable=unused-import
import warnings

from dotenv import load_dotenv

load_dotenv()

# TODO Oleksandr: get rid of this warning suppression when PromptLayer doesn't produce "Expected Choice but got dict"
#  warning anymore
warnings.filterwarnings("ignore", module="pydantic")

from try_gaia.single_question import main

if __name__ == "__main__":
    asyncio.run(main())
