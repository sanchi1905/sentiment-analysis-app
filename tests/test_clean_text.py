import re
from pathlib import Path

import pytest

import app


def test_clean_text_basic():
    # basic cleaning: remove punctuation, lowercase, and remove stopwords
    s = "Hello, world! This is a test."
    cleaned = app.clean_text(s)
    # 'this' and 'is' are stopwords and should be removed
    assert 'hello' in cleaned
    assert 'world' in cleaned
    assert 'this' not in cleaned
    assert 'is' not in cleaned


def test_clean_text_handles_none():
    assert app.clean_text(None) == ''
