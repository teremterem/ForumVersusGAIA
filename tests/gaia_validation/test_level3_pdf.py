# pylint: disable=import-outside-toplevel
"""
GAIA validation set. Level 3 samples that involve finding and reading PDF files.
"""

import pytest


@pytest.mark.asyncio
async def test_sea_star_fed_to_shrimp():
    """
    Test that the question about the sea star fed to the shrimp is answered correctly.
    """
    from forum_versus_gaia.gaia_agent import arun_assistant

    question = (
        "What integer-rounded percentage of the total length of the harlequin shrimp recorded in Omar "
        "Valencfia-Mendez 2017 paper was the sea star fed to the same type of shrimp in G. Curt Fiedler's 2002 "
        "paper?"
    )
    answer = await arun_assistant(question)
    assert answer == "22"
