"""Normalize the existing lab report for the nature-paper2word export engine."""

import json
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "lab1/docs"

TABLE_CAPTIONS = [
    "Experiment groups and original measurement environments",
    "CIFAR-10 architecture stages and parameter counts",
    "Original baseline performance under the equal 30-epoch budget",
    "Controlled AlexNet capacity and regularization configurations",
    "Validation-selected Task 3 results and Tesla T4 computational costs",
    "Paired seed-42 ResNet accuracy with and without training augmentation",
    "Prediction metrics for the two frozen baseline checkpoints",
    "FGSM and equal-budget random-control results on all 10,000 test images",
    "Corruption and adversarial accuracy for the same frozen baseline weights",
    "Held-out shared-class CIFAR-10 to STL-10 generalization",
    "Tesla T4 training costs and predictive performance for the five-run matrix",
    "FP32 inference performance measured on the Jade RTX 2000 Ada GPU",
]

WIDTHS = [[25, 25, 30, 30], [22, 52, 32, 16], [42, 18, 18], [30, 20, 16, 20],
          [28, 20, 25, 14, 17, 17, 20], [40, 22, 22, 24], [30, 23, 22, 20, 20, 20],
          [18, 18, 24, 24, 24, 24], [55, 24, 24], [40, 20, 20, 20, 20, 20],
          [45, 20, 22, 22, 22, 25], [48, 28, 35, 25]]

ABSTRACT = (
    "We evaluated CIFAR-10 adaptations of AlexNet and ResNet-18 through 13 training runs "
    "with a fixed train/validation split and a 30-epoch budget per run. Controlled AlexNet "
    "experiments varied convolutional width, Dropout and L2 regularization. Halving width "
    "reduced training time by 50.94% at a cost of 2.14 percentage points of test accuracy. "
    "For a separately trained seed-42 ResNet pair, crop-and-flip augmentation increased clean "
    "accuracy from 86.58% to 93.08% and mean fixed-corruption accuracy from 76.09% to 83.22%, "
    "while reducing Gaussian-noise accuracy. Frozen original checkpoints were much more "
    "sensitive to FGSM than to random perturbations under the same pixel budget. The "
    "augmented ResNet achieved 81.08% masked nine-class accuracy on held-out STL-10. "
    "Same-device cost measurements favour augmented ResNet for clean and geometric accuracy "
    "under the full training budget, and AlexNet when latency dominates. These conclusions "
    "remain specific to the measured checkpoints, corruption suite and hardware groups."
)


def main():
    original = (DOCS / "lab1_report.md").read_text(encoding="utf-8")
    body = original[original.index("## Experimental overview"):]
    body = body[:body.index("The course original is the submission standard:")]
    operational = body[body.index("The primary deliverable is"):body.index("| Evidence group")].strip()
    body = body.replace(operational + "\n\n", "", 1)
    body = body.replace("adaptations of AlexNet and ResNet-18,", "adaptations of AlexNet and ResNet-18 [@cifar; @alexnet; @resnet],", 1)
    body = body.replace("following the course Part A9 adaptations.", "following the course Part A9 adaptations. [@pytorch; @course]", 1)
    body = body.replace("[Srivastava et al. (2014)](https://jmlr.org/papers/v15/srivastava14a.html)", "Srivastava et al. [@dropout]")
    body = body.replace("[Goodfellow, Shlens and Szegedy](https://arxiv.org/abs/1412.6572)", "Goodfellow, Shlens and Szegedy [@fgsm]")
    body = body.replace("No STL labels were used for fitting or selection.", "No STL labels were used for fitting or selection. [@stl10]", 1)
    body = re.sub(r"\\\[\s*(.*?)\s*\\\]", lambda m: "$$" + " ".join(m[1].splitlines()) + "$$ {#eq-fgsm}", body, flags=re.S)
    body = body.replace("We tested budgets of 0,", "The attack in @eq-fgsm was evaluated at budgets of 0,", 1)
    body += "\n\n" + operational + "\n\nThe course assignment defines the submission scope [@assignment]. The accompanying team repository supplies the implementation and frozen experiment contract [@repository].\n"
    blocks = re.split(r"\n\s*\n", body.strip())
    output = ["---\nbibliography: lab1_references.bib\ncsl: nature.csl\nlink-citations: true\n---",
              "%% Title Accuracy robustness and computational tradeoffs of CIFAR 10 convolutional networks",
              "%% Subtitle Scalable Machine Learning Lab 1",
              "%% Author Lab 1 project team", "%% Abstract Title Abstract", "%% Abstract " + ABSTRACT,
              "%% Keywords Keywords: CIFAR-10; convolutional neural networks; regularization; robustness; adversarial examples"]
    main_number = 0
    sub_number = 0
    figure_number = 0
    table_number = 0
    for block in blocks:
        block = block.strip()
        if block.startswith("## ") and not block.startswith("### "):
            main_number += 1
            sub_number = 0
            heading = block.removeprefix("## ")
            heading = {"Experimental overview": "Experimental design",
                       "Reproduction, validation and references": "Reproducibility and artifact availability"}.get(heading, heading)
            output.append(f"# {main_number} {heading}")
        elif block.startswith("### "):
            sub_number += 1
            output.append(f"## {main_number}.{sub_number} " + block.removeprefix("### "))
        elif block.startswith("| "):
            table_number += 1
            lines = block.splitlines()
            width = WIDTHS[table_number - 1]
            lines[1] = "|" + "|".join("-" * value for value in width) + "|"
            output.append(f"%% Table Caption **Table {table_number}.** {TABLE_CAPTIONS[table_number - 1]}")
            output.append("\n".join(lines))
        elif block.startswith("!["):
            figure_number += 1
            caption, path = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", block).groups()
            with Image.open(DOCS / path) as image:
                width = min(15.7, 14.2 * image.width / image.height)
            for previous in range(len(output) - 1, -1, -1):
                if output[previous].startswith(("%%", "#", "|", "![", "![]", "$$")):
                    continue
                paragraph = output[previous]
                match = re.search(r" \(Figs?\. ([\d, and]+)\)\.$", paragraph)
                if match:
                    numbers = re.findall(r"\d+", match[1]) + [str(figure_number)]
                    label = ", ".join(numbers[:-1]) + " and " + numbers[-1]
                    output[previous] = paragraph[:match.start()] + f" (Figs. {label})."
                else:
                    output[previous] = paragraph.rstrip(".") + f" (Fig. {figure_number})."
                break
            output.append(f"![]({path}){{width={width:.2f}cm}}")
            output.append(f"%% Image Caption **Figure {figure_number}.** {caption}.")
        elif block.startswith("$$"):
            output.append(block)
        else:
            output.append(" ".join(block.splitlines()))
    output.append("%% References Title References")
    assert table_number == len(TABLE_CAPTIONS)
    target = DOCS / "lab1_report_paper.md"
    target.write_text("\n\n".join(output) + "\n", encoding="utf-8")
    original_rows = [line for line in original.splitlines() if line.startswith("|") and not re.fullmatch(r"[|: -]+", line)]
    final_rows = [line for line in target.read_text(encoding="utf-8").splitlines() if line.startswith("|") and not re.fullmatch(r"[|: -]+", line)]
    assert original_rows == final_rows, "Experimental table data changed"
    audit = {"table_count": table_number, "figure_count": figure_number,
             "original_table_rows_preserved": len(original_rows),
             "preset": "en-nature", "citation_style": "nature.csl"}
    (ROOT / ".runtime/paper_format_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
