"""Build the submission notebook and a source/artifact archive, excluding weights/data."""

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build_notebook(source_ref):
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True),
                "id": f"submission-{len(cells):02d}"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    report = (ROOT / "lab1/docs/lab1_report.md").read_text(encoding="utf-8")
    sections = re.split(r"(?=^## )", report, flags=re.M)
    add("markdown", sections[0] + "\nThis is an executed integration and presentation notebook. "
        "The 13 original training runs are preserved in the supplied artifacts. "
        "The following cells audit those records and render the final results; they do not rerun GPU training.\n")
    add("code", f'''from pathlib import Path
import subprocess
import sys

SOURCE_REF = {source_ref!r}
COLAB = Path('/content').exists()
if COLAB:
    ROOT = Path('/content/ScalableMlLab_submission')
    if not (ROOT / '.git').exists():
        subprocess.check_call(['git', 'clone', '--no-checkout',
                               'https://github.com/Mrpooool/ScalableMlLab.git', str(ROOT)])
    subprocess.check_call(['git', '-C', str(ROOT), 'fetch', 'origin', SOURCE_REF])
    subprocess.check_call(['git', '-C', str(ROOT), 'checkout', '--detach', SOURCE_REF])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           '-r', str(ROOT / 'lab1/requirements.txt')])
else:
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                if (p / 'lab1/src/submission.py').exists())
sys.path.insert(0, str(ROOT / 'lab1/src'))
from submission import audit_results, show_task, show_implementation
print('Pinned source revision:', SOURCE_REF)
print('Saved GPU measurements will be audited and displayed; no GPU training is requested.')
audit_results(ROOT)
''')
    task_map = {"Task 1:": "1", "Task 2:": "2", "Task 3:": "3", "Task 4:": "4",
                "Task 5:": "5", "Task 6:": "6", "Challenge 2:": "challenge2"}
    for section in sections[1:]:
        # Images are displayed by the next code cell and embedded in its actual outputs.
        add("markdown", re.sub(r"^!\[[^\]]*\]\([^\n]+\)\n?", "", section, flags=re.M))
        for prefix, task in task_map.items():
            if section.startswith("## " + prefix):
                add("code", f"show_task(ROOT, {task!r})\n")
                break
    add("markdown", "## Full implementation and frozen configuration appendix\n\n"
        "The following executed cell includes the complete supplied Python implementation "
        "and settings with file hashes. Exact split IDs are in the accompanying archive. "
        "Full training commands and weight locations are in `submission_instructions.md`.\n")
    add("code", "show_implementation(ROOT)\n")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                             "language": "python"},
                             "language_info": {"name": "python"},
                             "submission_source_ref": source_ref}}
    path = ROOT / "lab1/notebooks/lab1_submission.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path)


def source_archive():
    from sys import path as search_path
    search_path.insert(0, str(ROOT / "lab1/src"))
    from io_utils import aggregate_results_to_csv, save_history_csv
    rows = aggregate_results_to_csv(ROOT / "lab1/results", ROOT / "lab1/results/submission_summary.csv")
    for row in rows:
        record = json.loads((ROOT / "lab1/results" / f"{row['exp_id']}.json").read_text())
        source_id = record.get("checkpoint_ref", {}).get("source_exp_id")
        training_record = (json.loads((ROOT / "lab1/results" / f"{source_id}.json").read_text())
                           if source_id else record)
        row["training_device_name"] = training_record["environment"]["device_name"]
        row["evaluation_device_name"] = record.get("inference_benchmark", {}).get(
            "device_name", record["environment"]["device_name"])
        row["result_type"] = "frozen_checkpoint_evaluation" if source_id else "training_and_evaluation"
    save_history_csv(rows, ROOT / "lab1/results/submission_summary.csv")
    paths = []
    for directory in ["src", "configs", "tests", "results", "scripts", "docs"]:
        paths.extend(p for p in (ROOT / "lab1" / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts
                     and "smoke_role_b" not in p.parts
                     and p.suffix in {".py", ".json", ".csv", ".npz", ".png", ".md", ".tex", ".pdf"})
    paths.extend([ROOT / "AGENTS.md", ROOT / "Lab Assignment 1.docx.md",
                  ROOT / "lab1/requirements.txt"])
    paths.extend(ROOT / "lab1/notebooks" / name for name in [
        "role_a_task1_task2.ipynb", "role_b_task3_task5.ipynb", "role_b_reproduce.ipynb",
        "role_c_task4_challenge2_task6.ipynb"])
    paths = sorted(set(paths))
    inventory = {str(p.relative_to(ROOT).as_posix()): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in paths}
    inventory_path = ROOT / "lab1/submission_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    archive_path = ROOT / "lab1/submission_source.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths + [inventory_path]:
            archive.write(path, path.relative_to(ROOT))
    print(archive_path, f"({archive_path.stat().st_size:,} bytes)")


def package():
    names = ["notebooks/lab1_submission.ipynb", "docs/lab1_report.pdf", "docs/lab1_report.md",
             "docs/submission_instructions.md", "results/submission_validation.json",
             "submission_inventory.json", "submission_source.zip"]
    path = ROOT / "lab1/lab1_submission.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            source = ROOT / "lab1" / name
            archive.write(source, source.name)
    print(path, f"({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ref", help="Immutable source commit for the notebook bootstrap.")
    parser.add_argument("--package", action="store_true", help="Package the already executed notebook.")
    args = parser.parse_args()
    if args.source_ref:
        build_notebook(args.source_ref)
    source_archive()
    if args.package:
        package()
