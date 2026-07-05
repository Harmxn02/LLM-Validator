import json

import pytest

from util.validation import categorize_error, summarise_validation


@pytest.mark.parametrize(
    "message,expected",
    [
        (
            "Non-space characters found without seeing a doctype first. Expected “<!DOCTYPE html>”.",
            "missing-doctype",
        ),
        (
            "Element “head” is missing a required instance of child element “title”.",
            "missing-title",
        ),
        ("Stray doctype.", "stray-doctype"),
        ("Stray end tag “head”.", "stray-tag"),
        (
            "Start tag “body” seen but an element of the same type was already open.",
            "duplicate-tag",
        ),
        (
            "Element “title” not allowed as child of element “body” in this context.",
            "disallowed-child",
        ),
        (
            "Attribute “charset” not allowed on element “meta” at this point.",
            "disallowed-attribute",
        ),
        (
            "Element “meta” is missing one or more of the following attributes: “content”.",
            "missing-attribute",
        ),
        (
            "Cannot recover after last error. Any further errors will be ignored.",
            "cannot-recover",
        ),
        ("Non-space character in page trailer.", "trailing-content"),
        (
            'This document appears to be written in English. Consider adding “lang="en"”',
            "missing-lang",
        ),
        ("Some totally unrecognized validator message", "other"),
    ],
)
def test_categorize_error(message, expected):
    assert categorize_error(message) == expected


def test_summarise_validation(tmp_path):
    data = {
        "messages": [
            {"type": "error", "message": "Stray end tag “head”."},
            {
                "type": "info",
                "subType": "warning",
                "message": 'Consider adding lang="en"',
            },
            {"type": "info", "message": "Some info-only message"},
        ]
    }
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(data))

    summary = summarise_validation(str(path))

    assert summary["errors"] == 1
    assert summary["warnings"] == 1
    assert summary["infos"] == 1
    assert summary["categories"]["stray-tag"] == 1
    assert summary["categories"]["missing-lang"] == 1


def test_summarise_validation_empty_messages(tmp_path):
    path = tmp_path / "validation.json"
    path.write_text(json.dumps({"messages": []}))

    summary = summarise_validation(str(path))

    assert summary == {"errors": 0, "warnings": 0, "infos": 0, "categories": {}}
