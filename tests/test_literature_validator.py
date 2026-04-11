"""Tests for literature validation — all mocked, no network or API calls."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mozzarellm.clients.pmc_client import LiteratureClient
from mozzarellm.pipeline.literature_mcp import (
    _extract_genes_to_validate,
    _parse_json_from_text,
    validate_and_amend_with_mcp,
    validate_and_amend_without_mcp,
)


def test_extract_genes_to_validate():
    cluster = {
        "novel_role_genes": [{"gene": "HK2", "priority": 9}, {"gene": "DYRK1A", "priority": 7}],
        "uncharacterized_genes": [{"gene": "RBIS", "priority": 8}],
    }
    genes = _extract_genes_to_validate(cluster)
    assert set(genes) == {"HK2", "DYRK1A", "RBIS"}


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"key": "value"}', {"key": "value"}),
        ('```json\n{"key": "value"}\n```', {"key": "value"}),
        ("garbage text without json", None),
    ],
)
def test_parse_json_from_text(text, expected):
    result = _parse_json_from_text(text)
    assert result == expected


def test_mode_b_no_network():
    fake_response = {
        "resultList": {
            "result": [
                {"title": "HK2 in ribosomes", "pubYear": "2024", "doi": "10.1/x", "abstractText": "HK2 regulates rRNA"},
                {"title": "Another paper", "pubYear": "2023", "doi": "10.1/y", "abstractText": "DYRK1A study"},
            ]
        }
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_response
    mock_resp.raise_for_status.return_value = None

    with patch("requests.Session.get", return_value=mock_resp):
        client = LiteratureClient(cache_path=None)
        results = client.search_gene_pathway_literature(["HK2"], "ribosome biogenesis")

    assert len(results) == 2
    for r in results:
        for key in ("title", "year", "doi", "genes_mentioned"):
            assert key in r


def test_mode_a_skips_empty_cluster():
    cluster = {
        "dominant_process": "ribosome biogenesis",
        "novel_role_genes": [],
        "uncharacterized_genes": [],
    }

    with patch("anthropic.Anthropic") as mock_anthropic:
        result = validate_and_amend_with_mcp(cluster)

    assert "skipped" in result["_validation_metadata"]
    mock_anthropic.assert_not_called()


def test_mode_b_cost_estimate():
    fake_content = MagicMock()
    fake_content.text = (
        '{"dominant_process": "test", "novel_role_genes": [{"gene": "HK2"}], "uncharacterized_genes": []}'
    )

    fake_response = MagicMock()
    fake_response.usage.input_tokens = 1000
    fake_response.usage.output_tokens = 500
    fake_response.content = [fake_content]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        result = validate_and_amend_without_mcp(
            {
                "dominant_process": "test",
                "novel_role_genes": [{"gene": "HK2"}],
                "uncharacterized_genes": [],
            },
            [],
        )

    # (1000*3 + 500*15) / 1_000_000 = 0.0105
    assert result["_validation_metadata"]["cost_usd"] == pytest.approx(0.0105, abs=0.001)


def test_mode_a_cost_estimate():
    fake_content = MagicMock()
    fake_content.type = "text"
    fake_content.text = (
        '{"dominant_process": "test", "novel_role_genes": [{"gene": "HK2"}], "uncharacterized_genes": []}'
    )

    fake_response = MagicMock()
    fake_response.usage.input_tokens = 2000
    fake_response.usage.output_tokens = 800
    fake_response.content = [fake_content]

    mock_client = MagicMock()
    mock_client.beta.messages.create.return_value = fake_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        result = validate_and_amend_with_mcp(
            {
                "dominant_process": "test",
                "novel_role_genes": [{"gene": "HK2"}],
                "uncharacterized_genes": [],
            }
        )

    # (2000*3 + 800*15) / 1_000_000 = 0.018
    assert result["_validation_metadata"]["cost_usd"] == pytest.approx(0.018, abs=0.001)
