"""Audit the WWW 2027 review PDF and the local ACM template snapshot."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "CREDIT_WWW2027_v4.tex"
PDF = ROOT / "build/CREDIT_WWW2027_v4.pdf"
LOG = ROOT / "build/CREDIT_WWW2027_v4.log"
CLASS = ROOT / "acmart.cls"
BST = ROOT / "ACM-Reference-Format.bst"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def command(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    source = SOURCE.read_text()
    log = LOG.read_text(errors="replace")
    class_text = CLASS.read_text(errors="replace")
    pdf_info = command("pdfinfo", str(PDF))
    pdf_text = command("pdftotext", str(PDF), "-")
    first_page = command("pdftotext", "-f", "1", "-l", "1", str(PDF), "-")
    font_output = command("pdffonts", str(PDF))

    class_line = r"\documentclass[sigconf, anonymous, review]{acmart}"
    require(source.count(class_line) == 1, "official WWW 2027 class options differ")
    require("2026/08/16 v2.20" in class_text, "local acmart is not version 2.20")
    require(r"\settopmatter{printacmref=true}" in source, "ACM reference strip is disabled")
    for identifying_macro in (r"\author", r"\affiliation", r"\email", r"\orcid"):
        require(identifying_macro not in source, f"anonymous source contains {identifying_macro}")
    require("May 10--14, 2027" in source and "Dublin, Ireland" in source,
            "conference metadata does not match WWW 2027")
    require("agentic Web systems" in source,
            "source lacks an explicit first-page Web-relevance statement")
    figures = len(re.findall(r"\\begin\{figure\*?\}", source))
    descriptions = len(re.findall(r"\\Description\{", source))
    require(descriptions >= figures, "one or more figures lack an accessibility description")

    page_match = re.search(r"^Pages:\s+(\d+)$", pdf_info, re.MULTILINE)
    size_match = re.search(r"^Page size:\s+([^\n]+)$", pdf_info, re.MULTILINE)
    require(page_match is not None, "pdfinfo did not report page count")
    pages = int(page_match.group(1))
    page_size = size_match.group(1).strip() if size_match else "unknown"
    author_match = re.search(r"^Author:\s*(.*)$", pdf_info, re.MULTILINE)
    pdf_author = author_match.group(1).strip() if author_match else ""
    require(not pdf_author or "Anonymous" in pdf_author,
            "PDF metadata exposes a non-anonymous author")
    first_eight = command(
        "pdftotext", "-f", "1", "-l", str(min(8, pages)), str(PDF), "-"
    )
    require(pages <= 12, "PDF exceeds the 12-page hard limit")
    require("612 x 792 pts (letter)" in page_size, "PDF is not US Letter")
    require("Conclusion" in first_eight,
            "main-paper conclusion does not appear within the first eight pages")
    require("References" in pdf_text, "PDF does not contain references")

    lower_log = log.lower()
    require("undefined reference" not in lower_log, "compile log has an undefined reference")
    require("undefined citation" not in lower_log, "compile log has an undefined citation")
    require("latex error" not in lower_log, "compile log has a LaTeX error")
    overfull = [float(value) for value in re.findall(
        r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log
    )]
    max_overfull = max(overfull, default=0.0)
    require(max_overfull < 10.0, "compile log contains a visibly severe overfull line")

    font_lines = [line for line in font_output.splitlines()[2:] if line.strip()]
    require(font_lines, "pdffonts reported no fonts")
    require(all(" yes " in f" {line} " for line in font_lines), "PDF has an unembedded font")
    require(all("Type 3" not in line for line in font_lines), "PDF contains a Type 3 font")

    require("Anonymous Author(s)" in first_page, "review PDF is not anonymous")
    require("agentic Web systems" in first_page, "first page lacks an explicit Web relevance statement")
    require("/home/" not in pdf_text and "@" not in pdf_text,
            "PDF text exposes a local path or email address")
    require("Online calls" in source and "1,828,683 physical pre-test calls" in source,
            "online and offline call costs are not separated")

    protected = {
        "AGENTS.md": "33168b848f700c1451620891dbbec34c3a2eef3c1bbc31391a4da0aec4e21cdf",
        "CREDIT_WWW2027_v3.tex": "31e5fb029a546ea0a21dd74b343804c2ea1ade6a17c501a1875454ad0a88d153",
    }
    for relative, expected in protected.items():
        require(digest(ROOT / relative) == expected, f"protected file changed: {relative}")

    report = {
        "verified": True,
        "checked_on": "2026-10-19",
        "conference": "The ACM Web Conference 2027 Research Track",
        "official_class_options": class_line,
        "template": {
            "acmart_version": "2.20 (2026-08-16)",
            "acmart_sha256": digest(CLASS),
            "bibliography_style_sha256": digest(BST),
        },
        "pdf": {
            "pages": pages,
            "total_page_limit": 12,
            "main_content_limit": 8,
            "main_conclusion_within_first_eight": True,
            "page_size": page_size,
            "anonymous": True,
            "anonymous_metadata": True,
            "web_relevance_on_first_page": True,
            "figure_descriptions": descriptions,
            "fonts_embedded": True,
            "type3_fonts": False,
            "font_count": len(font_lines),
            "unresolved_references": 0,
            "max_overfull_hbox_pt": round(max_overfull, 3),
            "sha256": digest(PDF),
        },
        "source_sha256": digest(SOURCE),
        "protected_files_sha256": protected,
        "verifier_sha256": digest(Path(__file__)),
        "requirements_url": "https://www2027.thewebconf.org/research-track-papers/",
        "template_url": "https://ctan.org/pkg/acmart?lang=en",
    }
    output = ROOT / "results/credit_v4_www2027_format_audit_v1.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
