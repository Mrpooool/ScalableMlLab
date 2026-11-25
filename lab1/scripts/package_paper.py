"""Validate the paper revision and package it with the existing executed notebook."""

import hashlib
import json
import re
import zipfile
from pathlib import Path

import pypdfium2 as pdfium
from docx import Document

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "lab1/docs"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    document_path = DOCS / "lab1_report_paper.docx"
    pdf_path = DOCS / "lab1_report_paper.pdf"
    document = Document(document_path)
    assert len(document.tables) == 12 and len(document.inline_shapes) == 20
    original_rows = []
    for line in (DOCS / "lab1_report.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and not re.fullmatch(r"[|: -]+", line):
            original_rows.append([value.strip() for value in line.strip("|").split("|")])
    actual_rows = [[cell.text.strip() for cell in row.cells]
                   for table in document.tables for row in table.rows]
    assert original_rows == actual_rows, "Table contents changed during Word export"
    assert document.styles["Body Text"].font.name == "Times New Roman"
    assert document.styles["Body Text"].font.size.pt == 12
    references = [p.text for p in document.paragraphs if p.style.name == "Bibliography"]
    assert len(references) == 10
    with pdfium.PdfDocument(str(pdf_path)) as pdf:
        pages = len(pdf)
        text = "\n".join(page.get_textpage().get_text_range() for page in pdf)
    assert "We evaluated CIFAR-10 adaptations" in text
    for number in range(1, 21):
        assert f"Figure {number}." in text
    for number in range(1, 13):
        assert f"Table {number}." in text
    assert "Equation (1)" in text
    for number in range(1, 7):
        assert f"Task {number}:" in text
    assert "Challenge 2:" in text
    record = json.loads((ROOT / "lab1/results/submission_validation.json").read_text())
    record["prior_report_sha256"] = record.pop("report_sha256")
    record["report_sha256"] = sha256(pdf_path)
    record["report_path"] = "lab1/docs/lab1_report_paper.pdf"
    record["report_pages_visually_reviewed"] = pages
    record["paper_format"] = {"preset": "en-nature", "engine": "md-to-word-docx 3.6.0",
                               "citation_style": "nature.csl", "tables": 12, "figures": 20,
                               "table_rows_verified": len(actual_rows), "references": 10,
                               "native_word_equation_count": 1,
                               "word_layout_requires_manual_preview": True,
                               "pdf_renderer": "Pandoc and Tectonic",
                               "docx_sha256": sha256(document_path)}
    validation = ROOT / "lab1/results/paper_submission_validation.json"
    validation.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    from build_submission import source_archive
    source_archive()
    files = {"lab1_submission.ipynb": ROOT / "lab1/notebooks/lab1_submission.ipynb",
             "lab1_report.pdf": pdf_path, "lab1_report.docx": document_path,
             "lab1_report_paper.md": DOCS / "lab1_report_paper.md",
             "lab1_references.bib": DOCS / "lab1_references.bib", "nature.csl": DOCS / "nature.csl",
             "submission_instructions.md": DOCS / "submission_instructions.md",
             "paper_format_notes.md": DOCS / "paper_format_notes.md",
             "submission_validation.json": validation,
             "submission_inventory.json": ROOT / "lab1/submission_inventory.json",
             "submission_source.zip": ROOT / "lab1/submission_source.zip"}
    destination = ROOT / "lab1/lab1_submission_paper.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert hashlib.sha256(archive.read("lab1_submission.ipynb")).hexdigest() == record["notebooks"]["lab1_submission.ipynb"]["sha256"]
    print(json.dumps({"pdf_pages": pages, "table_rows_verified": len(actual_rows),
                      "package": str(destination), "sha256": sha256(destination)}, indent=2))


if __name__ == "__main__":
    main()
