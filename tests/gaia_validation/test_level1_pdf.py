# pylint: disable=import-outside-toplevel
"""
GAIA validation set. Level 1 samples that involve finding and reading PDF files.
"""

import pytest


@pytest.mark.asyncio
async def test_dragons_diet():
    """
    Test that the dragons diet question is answered correctly.
    """
    from forum_versus_gaia.gaia_agent import arun_assistant

    question = (
        "What was the volume in m^3 of the fish bag that was calculated in the University of Leicester paper "
        '"Can Hiccup Supply Enough Fish to Maintain a Dragon’s Diet?"'
    )
    answer = await arun_assistant(question)
    assert answer == "0.1777"


@pytest.mark.asyncio
async def test_doctor_who_location():
    """
    Test that the Doctor Who location question is answered correctly.
    """
    from forum_versus_gaia.gaia_agent import arun_assistant

    question = (
        "In Series 9, Episode 11 of Doctor Who, the Doctor is trapped inside an ever-shifting maze. What is this "
        "location called in the official script for the episode? Give the setting exactly as it appears in the "
        "first scene heading."
    )
    answer = await arun_assistant(question)
    # The instruction "give the setting exactly as it appears in the first scene heading" is somewhat misleading,
    # which results in the answer sometimes containing not only "THE CASTLE" but other words as well - no point in
    # trying to fix this.
    assert "THE CASTLE" in answer


@pytest.mark.asyncio
async def test_nasa_award():
    """
    Test that the question about NASA award number for R. G. Arendt's work is answered correctly.
    """
    from forum_versus_gaia.gaia_agent import arun_assistant

    question = (
        "On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. This article "
        "mentions a team that produced a paper about their observations, linked at the bottom of the article. "
        "Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?"
    )
    answer = await arun_assistant(question)
    assert answer == "80GSFC21M0002"


@pytest.mark.asyncio
async def test_bert_base_blocks():
    """
    Test that the question about the number of blocks in BERT base encoder vs the encoder from the architecture
    proposed in Attention is All You Need is answered correctly.
    """
    from forum_versus_gaia.gaia_agent import arun_assistant

    question = (
        "How many more blocks (also denoted as layers) in BERT base encoder than the encoder from the "
        "architecture proposed in Attention is All You Need?"
    )
    answer = await arun_assistant(question)
    assert answer == "6"
