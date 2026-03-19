"""Tests that example scripts remain runnable."""

from __future__ import annotations


def test_custom_reader_example() -> None:
    """Ensure the custom reader example runs without errors."""
    from examples.custom_reader_example import main

    main()
