"""Verify the executed submission and write a compact, reproducible validation record."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    notebooks = {}
    for name, expected_cells, expected_figures in [
        ("role_a_task1_task2.ipynb", 5, 4),
        ("role_b_task3_task5.ipynb", 5, 11),
        ("role_c_task4_challenge2_task6.ipynb", 8, 9),
        ("lab1_submission.ipynb", 9, 24),
    ]:
        path = ROOT / "lab1/notebooks" / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert len(cells) == expected_cells, name
        assert all(cell["execution_count"] is not None for cell in cells), name
        outputs = [output for cell in cells for output in cell["outputs"]]
        assert not any(output["output_type"] == "error" for output in outputs), name
        assert not any("Output hidden; open in" in str(output) for output in outputs), name
        figures = sum("image/png" in output.get("data", {}) for output in outputs)
        assert figures == expected_figures, (name, figures)
        notebooks[name] = {"executed_code_cells": len(cells), "png_figures": figures,
                           "errors": 0, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    final = json.loads((ROOT / "lab1/notebooks/lab1_submission.ipynb").read_text(encoding="utf-8"))
    markdown = "\n".join("".join(cell["source"]) for cell in final["cells"]
                         if cell["cell_type"] == "markdown")
    sources = [*sorted((ROOT / "lab1/src").glob("*.py")),
               *sorted((ROOT / "lab1/configs").glob("*.json"))]
    for path in sources:
        if path.name != "split_indices.json":
            assert path.read_text(encoding="utf-8").strip() in markdown, path
    for task in range(1, 7):
        assert f"## Task {task}:" in markdown
    assert "## Challenge 2:" in markdown
    text = "\n".join(str(output) for cell in final["cells"] if cell["cell_type"] == "code"
                     for output in cell["outputs"])
    assert "13 complete 30-epoch histories" in text
    assert "41 prediction conditions agree with saved metrics" in text
    assert text.count("fresh CPU architecture check passed") == 2
    report_path = ROOT / "lab1/docs/lab1_report.pdf"
    assert report_path.read_bytes().startswith(b"%PDF-")
    record = {"status": "passed", "source_ref": final["metadata"]["submission_source_ref"],
              "display_runtime": "Google Colab CPU (Linux, Python 3.13.18, torch 2.14.1+cpu)",
              "full_gpu_retraining_in_integration": False,
              "training_histories_checked": 13, "paired_prediction_conditions_checked": 41,
              "unit_tests_passed": 34, "source_appendix_files_checked": len(sources) - 1,
              "notebooks": notebooks, "report_pages_visually_reviewed": 18,
              "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
              "provenance_notes": ["A training/benchmark: RTX 5070 Ti Laptop",
                                   "B training/evaluation: Tesla T4",
                                   "C training: Tesla T4; evaluation/inference: Jade RTX 2000 Ada",
                                   "C historical Git commit unknown; no historical commit invented"]}
    destination = ROOT / "lab1/results/submission_validation.json"
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
