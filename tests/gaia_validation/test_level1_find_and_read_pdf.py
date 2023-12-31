"""
GAIA validation set. Level 1 samples that involve finding and reading PDF files.
"""
import pytest

from forum_versus_gaia.gaia_agent import arun_assistant


@pytest.mark.asyncio
async def test_dragons_diet():
    """
    Test that the dragons diet question is answered correctly.
    """
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
    question = (
        "In Series 9, Episode 11 of Doctor Who, the Doctor is trapped inside an ever-shifting maze. What is this "
        "location called in the official script for the episode? Give the setting exactly as it appears in the "
        "first scene heading."
    )
    answer = await arun_assistant(question)
    # TODO Oleksandr: fix the assistant so it only returns "THE CASTLE" and no other text
    assert "THE CASTLE" in answer
