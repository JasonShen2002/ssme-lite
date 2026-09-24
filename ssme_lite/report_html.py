"""Offline HTML renderer. Statistics are supplied by report_data, not the UI."""
from pathlib import Path


def render_report(payload):
    assets = Path(__file__).with_name("report_assets")
    template = (assets / "report.html").read_text(encoding="utf-8")
    # Escape raw-text element terminators and Unicode JavaScript separators.
    safe = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return (template.replace("/* REPORT_CSS */", (assets / "report.css").read_text(encoding="utf-8"))
            .replace("/* REPORT_JS */", (assets / "report.js").read_text(encoding="utf-8"))
            .replace("REPORT_DATA_JSON", safe))
