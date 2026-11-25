"""Build the reproducible Role B notebook and its small, complete source archive."""

import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
cells = []


def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "source": text.splitlines(True),
                  "execution_count": None, "outputs": []})


markdown("""# Lab 1 — Role B: Capacity, Regularization and Adversarial Analysis

This notebook executes Tasks 3 and 5 using the team's Python modules. The accompanying
`role_b_source.zip` contains the complete implementation and frozen configurations.
Data and model weights are kept outside Git. All figures and tables below are measured outputs.

Task 3 uses six predeclared AlexNet configurations: a common reference, half-width,
dropout 0/0.25/0.5, and SGD weight decay 0/1e-4/5e-4. Each comparison changes one factor.
Task 5 uses the team's frozen AlexNet and ResNet checkpoints; no model fitting occurs there.
The official test set is opened only after the experiment protocol has been fixed.
""")
code("""from pathlib import Path
import sys, subprocess, zipfile
COLAB = Path('/content').exists()
if COLAB:
    ROOT = Path('/content/ScalableMlLab')
    if not (ROOT / 'lab1/src/run_role_b_experiments.py').exists():
        from google.colab import files
        print('Upload the accompanying role_b_source.zip')
        uploaded = files.upload()
        with zipfile.ZipFile('role_b_source.zip') as archive:
            archive.extractall(ROOT)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           '-r', str(ROOT / 'lab1/requirements.txt')])
else:
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                if (p / 'lab1/src/run_role_b_experiments.py').is_file())
sys.path.insert(0, str(ROOT / 'lab1/src'))
print('Code root:', ROOT)
""")
markdown("""## Storage and hardware

In Colab, first select a T4 GPU when available. Add the team's shared folder to My Drive
and adjust `SHARED_ROOT` below to its actual shortcut path. The two Task 5 weights must
exist under `checkpoints/t2_alexnet_cifar_baseline_s42/best.pt` and
`checkpoints/t2_resnet18_cifar_baseline_s42/best.pt`.
Role B outputs are saved separately so A/C records are preserved.
""")
code("""import torch
from io_utils import get_environment_info, load_json
if COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    SHARED_ROOT = Path('/content/drive/MyDrive/ScalableMlLab/lab1')
    B_ROOT = Path('/content/drive/MyDrive/ScalableMlLab_role_b')
    CHECKPOINTS = B_ROOT / 'checkpoints'
    RESULTS = B_ROOT / 'results'
    import shutil
    for exp_id in load_json(ROOT / 'lab1/configs/role_b_experiments.json')['prediction_models']:
        destination = CHECKPOINTS / exp_id / 'best.pt'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(SHARED_ROOT / 'checkpoints' / exp_id / 'best.pt', destination)
else:
    CHECKPOINTS, RESULTS = ROOT / 'lab1/checkpoints', ROOT / 'lab1/results'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(get_environment_info(DEVICE))
if not COLAB:
    print('Local run: exclude its costs from cross-hardware T4/Jade rankings.')
""")
markdown("""## Protocol checks and smoke test

The unit tests check raw-pixel gradient propagation, perturbation bounds, untouched model
weights/gradients, the ASR denominator, per-ID random controls, and the controlled matrix.
The smoke run uses 128 training and 64 validation images for one epoch; its outputs are
isolated and are never substituted for full experiments.
""")
code("""subprocess.check_call([sys.executable, '-m', 'unittest', 'discover',
                       '-s', str(ROOT / 'lab1/tests'), '-p', 'test_role_b.py', '-v'])
from run_role_b_experiments import main
common = ['--data-root', str(ROOT / 'lab1/data'), '--checkpoint-root', str(CHECKPOINTS),
          '--results-dir', str(RESULTS), '--device', DEVICE, '--quiet']
main(['--stage', 'smoke', *common])
""")
markdown("""## Task 3 — Controlled training and final evaluation

Every configuration uses 45,000 training images, 5,000 validation images, seed 42,
batch 128, 30 epochs (1,350,000 trained samples), SGD momentum 0.9, cosine scheduling,
lr 0.01 and no augmentation. Validation accuracy selects the checkpoint; lower
validation CE breaks ties. Both FC hidden sizes remain 1024/512 when convolution width changes.
SGD decay applies to all model parameters, including biases. The plotted CE excludes the L2 term.

Completed runs are checked and reused; incomplete or differently configured runs are rejected.
The matrix and attack protocol were fixed before final test evaluation.
""")
code("""main(['--stage', 'train', *common])
main(['--stage', 'evaluate', *common])
""")
code("""import pandas as pd
from IPython.display import display, Image, Markdown
display(pd.read_csv(RESULTS / 'role_b_task3_summary.csv'))
for name in ['capacity', 'dropout', 'l2']:
    display(Image(filename=str(RESULTS / 'figures/role_b' / f't3_{name}.png')))
""")
markdown("""## Task 5 — Prediction diagnostics and adversarial inputs

Metrics cover all 10,000 test images. Confusion matrices are normalized within each true
class; example images are the first five correct and first five incorrect by image ID.
FGSM is untargeted, with raw-pixel L-infinity budgets 0, 2/255, 4/255 and 8/255.
Model evaluation mode is used while input gradients remain enabled during attack generation.
ASR counts newly incorrect predictions only among originally correct images.

Uniform random noise uses the same L-infinity budgets and per-image seeds. The separate
Task 4 corruptions use C's frozen per-ID protocol; their strengths are not matched to FGSM.
Adversarial examples show two first successes and one first failure when available.
The perturbation panel is amplified as `clip(0.5 + 8*delta, 0, 1)` for visibility.
""")
code("""display(pd.read_csv(RESULTS / 'role_b_task5_clean.csv'))
display(pd.read_csv(RESULTS / 'role_b_task5_attacks.csv'))
display(Image(filename=str(RESULTS / 'figures/role_b/t5_training_diagnostics.png')))
display(Image(filename=str(RESULTS / 'figures/role_b/t5_fgsm_vs_random.png')))
for source_id in load_json(ROOT / 'lab1/configs/role_b_experiments.json')['prediction_models']:
    exp_id = f't5_{source_id}'
    display(pd.read_csv(RESULTS / f'{exp_id}_classes.csv'))
    for suffix in ['confusion', 'predictions', 'fgsm']:
        display(Image(filename=str(RESULTS / 'figures/role_b' / f'{exp_id}_{suffix}.png')))
""")
markdown("""## Evidence and limitations

Use the accompanying English `role_b_report.md` for the measured interpretation.
Source-file/config hashes, checkpoint hashes and device metadata are stored in each result.
One seed supports controlled within-run comparisons; it does not establish multi-seed stability.
Thirty epochs may leave some variants incompletely optimized. A one-step FGSM attack
does not certify adversarial robustness. Hardware costs are comparable only within the
same hardware/protocol group; local GPU values must not be ranked against T4/Jade timings.

## References

- [Course assignment](https://github.com/Mrpooool/ScalableMlLab/blob/main/Lab%20Assignment%201.docx.md)
- [Course notebook, Part A9 and Part B](https://github.com/Mrpooool/ScalableMlLab/blob/main/lab1/notebooks/ID2223_Lab_Session_2.ipynb)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- [ResNet](https://arxiv.org/abs/1512.03385)
- [FGSM: Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
- [Team repository and reused Python modules](https://github.com/Mrpooool/ScalableMlLab)
""")

notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"}, "accelerator": "GPU",
}, "cells": cells}
for index, cell in enumerate(cells):
    cell["id"] = f"role-b-{index:02d}"
destination = ROOT / "lab1/notebooks/role_b_reproduce.ipynb"
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

archive_path = ROOT / "lab1/role_b_source.zip"
revision = subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}",
                                    "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
paths = [*ROOT.glob("lab1/src/*.py"), *ROOT.glob("lab1/configs/*.json"),
         *ROOT.glob("lab1/tests/*.py"), ROOT / "lab1/requirements.txt",
         ROOT / "AGENTS.md", ROOT / "lab1/docs/experiment_contract.md",
         *ROOT.glob("lab1/results/histories/t2_*_baseline_s42.csv"),
         ROOT / "lab1/results/checkpoints_manifest.json",
         *ROOT.glob("lab1/docs/role_b*.md")]
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.writestr("lab1/role_b_source_revision.json", json.dumps({"base_commit": revision}))
    for path in paths:
        archive.write(path, path.relative_to(ROOT))
print(destination)
print(archive_path)
