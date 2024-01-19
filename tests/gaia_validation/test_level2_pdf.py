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


@pytest.mark.skip  # TODO TODO TODO
@pytest.mark.asyncio
async def test_freud_neurologist():
    """
    Test that the question about the neurologist in a book about Freud is answered correctly.
    """
    question = (
        "The book with the doi 10.1353/book.24372 concerns a certain neurologist. According to chapter 2 of the "
        "book, what author influenced this neurologist’s belief in “endopsychic myths”? Give the last name only."
    )
    answer = await arun_assistant(question)
    assert answer == "Kleinpaul"


@pytest.mark.skip  # TODO TODO TODO
@pytest.mark.asyncio
async def test_elm_street():
    """
    Test that the question about the horror movie is answered correctly.
    """
    question = (
        "In Valentina Re’s contribution to the 2017 book “World Building: Transmedia, Fans, Industries”, what "
        "horror movie does the author cite as having popularized metalepsis between a dream world and reality? "
        "Use the complete name with article if any."
    )
    answer = await arun_assistant(question)
    assert answer == "A Nightmare on Elm Street"
