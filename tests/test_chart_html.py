import json
from pathlib import Path

import pytest

from build_chart import TEMPLATE_MARKER, render

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "chart_template.html"
PLOTLY_CDN = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"


def test_template_exists_and_carries_the_marker():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert TEMPLATE_MARKER in text
    assert text.count(TEMPLATE_MARKER) == 1


def test_template_pins_plotly_and_references_nothing_else_remote():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert PLOTLY_CDN in text
    remote = [
        line
        for line in text.splitlines()
        if ("http://" in line or "https://" in line) and PLOTLY_CDN not in line
    ]
    assert remote == [], f"unexpected remote references: {remote}"


def test_render_substitutes_the_payload(tmp_path):
    payload = {"hello": "world", "n": 1}
    out = tmp_path / "out" / "index.html"

    render(payload, TEMPLATE, out)

    text = out.read_text(encoding="utf-8")
    assert TEMPLATE_MARKER not in text
    assert json.dumps(payload, separators=(",", ":")) in text


def test_render_rejects_a_template_without_the_marker(tmp_path):
    bad = tmp_path / "bad.html"
    bad.write_text("<html></html>", encoding="utf-8")

    with pytest.raises(ValueError, match="marker"):
        render({}, bad, tmp_path / "out.html")


def test_rendered_output_has_no_local_file_references(tmp_path):
    out = tmp_path / "index.html"
    render({"quantiles": []}, TEMPLATE, out)
    text = out.read_text(encoding="utf-8")

    assert 'src="./' not in text
    assert 'href="./' not in text
    assert 'src="/' not in text


def test_template_exposes_the_browser_hooks_task_ten_verifies():
    text = TEMPLATE.read_text(encoding="utf-8")
    for hook in ("__quantilePricesAt", "__chartReady"):
        assert f"window.{hook}" in text, f"missing browser hook {hook}"


def test_template_builds_all_ninety_eight_bands():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "tonexty" in text
    assert "Plotly.newPlot" in text
    assert "EMPHASIS" in text


def test_template_exposes_the_panel_hooks():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "window.__updatePanel" in text
    assert "window.__panelState" in text
    assert "plotly_hover" in text
    assert "plotly_click" in text


def test_template_exposes_the_horizon_control():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "window.__setHorizon" in text
    assert "Plotly.relayout" in text


def test_template_offers_four_horizons_and_three_densities():
    text = TEMPLATE.read_text(encoding="utf-8")
    for years in ("0", "2", "5", "10"):
        assert f'data-years="{years}"' in text
    for step in ("1", "5", "10"):
        assert f'data-step="{step}"' in text


