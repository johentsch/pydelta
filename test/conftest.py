import os
from pathlib import Path

import pytest


@pytest.fixture(scope='session')
def testdir() -> str:
    return os.fspath(Path(__file__).parent / 'corpus3')