"""Validate and package only Role B artifacts; datasets and teammates' outputs stay out."""

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab1/results"
matrix = json.loads((ROOT / "lab1/configs/role_b_experiments.json").read_text())
manifest_path = RESULTS / "role_b_checkpoints_manifest.json"
manifest = json.loads(manifest_path.read_text())
expected = {f"t3_alexnet_{row['name']}_s42" for row in matrix["experiments"]}
if {entry["exp_id"] for entry in manifest["checkpoints"]} != expected:
    raise ValueError("Manifest must contain exactly the six frozen Task 3 configurations")
for entry in manifest["checkpoints"]:
    record = json.loads((RESULTS / f"{entry['exp_id']}.json").read_text())
    checkpoint = ROOT / "lab1" / entry["drive_rel_path"]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != record["role_b"]["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint changed: {entry['exp_id']}")
    if record["training"]["num_trained_samples"] != 1_350_000:
        raise ValueError(f"Incomplete training: {entry['exp_id']}")
    entry["config"] = record["config"]
    entry["result_file"] = f"lab1/results/{entry['exp_id']}.json"
    record["training"]["history_file"] = f"lab1/results/histories/{entry['exp_id']}.csv"
    (RESULTS / f"{entry['exp_id']}.json").write_text(json.dumps(record, indent=2) + "\n")
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

for source_id in matrix["prediction_models"]:
    record = json.loads((RESULTS / f"t5_{source_id}.json").read_text())
    attacks = record["evaluation"]["adversarial_metrics"]
    if len(attacks) != 8 or any(row["num_samples"] != 10_000 for row in attacks):
        raise ValueError(f"Incomplete Task 5: {source_id}")

artifact_paths = set()
for pattern in ("t3_*.json", "t5_*.json", "t5_*.csv", "role_b_*.json", "role_b_*.csv",
                "histories/t3_*.csv", "predictions/t5_*.npz", "figures/role_b/*"):
    artifact_paths.update(RESULTS.glob(pattern))
for pattern in ("lab1/src/*.py", "lab1/configs/*.json", "lab1/tests/*.py", "lab1/docs/role_b*.md",
                "lab1/scripts/*role_b*.py", "lab1/notebooks/role_b*.ipynb",
                "lab1/results/histories/t2_*_baseline_s42.csv"):
    artifact_paths.update(ROOT.glob(pattern))
artifact_paths.update([ROOT / "AGENTS.md", ROOT / "lab1/requirements.txt",
                       ROOT / "lab1/docs/experiment_contract.md", ROOT / "lab1/role_b_source.zip"])

with zipfile.ZipFile(ROOT / "role_b_results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(artifact_paths):
        if path.is_file():
            archive.write(path, path.relative_to(ROOT))
with zipfile.ZipFile(ROOT / "role_b_checkpoints.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for entry in manifest["checkpoints"]:
        path = ROOT / "lab1" / entry["drive_rel_path"]
        archive.write(path, entry["drive_rel_path"])
    archive.write(manifest_path, "role_b_checkpoints_manifest.json")
print("Validated six complete Task 3 runs and both full-test Task 5 evaluations.")
print(ROOT / "role_b_results.zip")
print(ROOT / "role_b_checkpoints.zip")
