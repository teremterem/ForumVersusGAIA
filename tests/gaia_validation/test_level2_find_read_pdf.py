"""
GAIA validation set. Level 2 samples that involve finding and reading PDF files.
"""
import pytest

from forum_versus_gaia.gaia_agent import arun_assistant


@pytest.mark.asyncio
async def test_bulgarian_gender_split():
    """
    Test that the question about the gender split in the 2011 Bulgarian census is answered correctly.
    """
    question = (
        "Take the gender split from the 2011 Bulgarian census about those who have completed tertiary education. "
        "Subtract the smaller number from the larger number, then return the difference in thousands of women. "
        'So if there were 30.1 thousand more men, you\'d give "30.1"'
    )
    answer = await arun_assistant(question)
    assert answer == "234.9"
