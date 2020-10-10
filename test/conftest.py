"""
A few fixtures usable for many tests.

To speed testing up, all of these fixtures are marked as 'session' fixtures.
Thus, they MUST NOT be modified by the tests.
"""

import os
from pathlib import Path

import pytest

import delta


@pytest.fixture(scope='session')
def testdir() -> str:
    """
    Test directory with 9 texts from 3 authors
    """
    return os.fspath(Path(__file__).parent / 'corpus3')


@pytest.fixture(scope='session')
def corpus(testdir) -> delta.Corpus:
    """
    Raw carpus built from the test directory.
    """
    return delta.Corpus(testdir)


@pytest.fixture(scope='session')
def c50(corpus) -> delta.Corpus:
    """
    Sample corpus limited to the 50 most frequent words.
    """
    c50 = corpus.get_mfw_table(50)
    return c50


@pytest.fixture(scope='session')
def distances(c50):
    """
    A sample distance matrix.
    """
    return delta.functions.cosine_delta(c50)
