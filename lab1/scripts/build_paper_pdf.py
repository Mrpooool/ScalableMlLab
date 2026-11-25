"""Render the same normalized paper through Pandoc and a supplied TeX engine."""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "lab1/docs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pandoc", default="pandoc")
    parser.add_argument("--tectonic", default="tectonic")
    args = parser.parse_args()
    source = (DOCS / "lab1_report_paper.md").read_text(encoding="utf-8")
    metadata_text, body = source[4:].split("\n---", 1)
    metadata = yaml.safe_load(metadata_text)
    metadata["bibliography"] = (DOCS / metadata["bibliography"]).as_posix()
    metadata["csl"] = (DOCS / "nature.csl").as_posix()
    body = re.sub(r"^%% Abstract Title .+$", "", body, flags=re.M)
    for style, key in [("Title", "title"), ("Subtitle", "subtitle"),
                       ("Author", "author"), ("Abstract", "abstract")]:
        pattern = rf"^%% {style} (.+)$"
        metadata[key] = re.search(pattern, body, flags=re.M)[1]
        body = re.sub(pattern, "", body, flags=re.M)
    body = re.sub(r"^%% Keywords (.+)$", r"**\1**", body, flags=re.M)
    body = re.sub(r"^%% Table Caption (.+)$", r"\1", body, flags=re.M)
    body = re.sub(r"!\[\]\(([^)]+)\)(\{[^}]+\})\s*%% Image Caption (.+)",
                  lambda m: f"![{m[3]}]({(DOCS / m[1]).resolve().as_posix()}){m[2]}", body)
    body = body.replace("@eq-fgsm", "Equation (1)")
    body = re.sub(r"\$\$(.+?)\$\$ \{#eq-fgsm\}",
                  lambda m: "\\begin{equation}\n" + m[1] + "\n\\end{equation}", body)
    body = body.replace("%% References Title References", "# References {.unnumbered}")
    preamble = [
        r"\usepackage{caption}", r"\captionsetup[figure]{labelformat=empty}",
        r"\usepackage{float}", r"\usepackage{needspace}",
        r"\setlength{\emergencystretch}{3em}",
        r"\makeatletter\setlength{\@fptop}{0pt}\setlength{\@fpsep}{14pt}\makeatother",
        r"\renewenvironment{abstract}{\par\noindent\textbf{Abstract}\par}{\par\medskip}",
    ]
    metadata["colorlinks"] = False
    build = ROOT / ".runtime/paper_typeset"
    build.mkdir(parents=True, exist_ok=True)
    markdown = build / "paper.md"
    markdown.write_text("---\n" + yaml.safe_dump(metadata, allow_unicode=True) + "---\n" + body, encoding="utf-8")
    tex = build / "lab1_report_paper.tex"
    subprocess.run([args.pandoc, str(markdown), "--standalone", "--citeproc", "-t", "latex",
                    "-V", "mainfont=Times New Roman", "-V", "fontsize=12pt", "-V", "papersize=a4",
                    "-V", "geometry:margin=2.54cm", "-V", "linestretch=1.5",
                    "-o", str(tex)], check=True)
    latex = tex.read_text(encoding="utf-8")
    latex = latex.replace(r"\begin{document}", "\n".join(preamble) + "\n" + r"\begin{document}")
    latex = re.sub(r"\\begin\{figure\}(?:\[[^\]]*\])?", r"\\begin{figure}[H]", latex)
    def keep_table(match):
        caption, table = match.groups()
        table = table.replace(r"\begin{longtable}[]", r"\begin{tabular}")
        table = table.replace("\\endhead\n\\bottomrule\\noalign{}\n\\endlastfoot", "")
        table = table.replace(r"\end{longtable}", "\\bottomrule\n\\end{tabular}")
        return ("\\begin{table}[H]\n\\begin{minipage}{\\linewidth}\n\\small\\singlespacing\n"
                + caption + "\\par\\vspace{6pt}\n" + table
                + "\n\\end{minipage}\n\\end{table}")
    latex = re.sub(r"(\\textbf\{Table \d+\.\}.*?)(?:\n\n)(\\begin\{longtable\}.*?\\end\{longtable\})",
                   keep_table, latex, flags=re.S)
    latex = latex.replace(r"\section*{References}", r"\clearpage\section*{References}")
    latex = latex.replace(r"\begin{CSLReferences}{0}{0}", r"\begingroup\small\setstretch{1.15}\begin{CSLReferences}{0}{0}")
    latex = latex.replace(r"\end{CSLReferences}", r"\end{CSLReferences}\endgroup")
    latex = re.sub(r"(?m)^(\d{1,2})\.(?=[A-Z\\])", r"\1.\\quad ", latex)
    tex.write_text(latex, encoding="utf-8")
    subprocess.run([args.tectonic, "--keep-logs", "--outdir", str(build), str(tex)], check=True)
    output = DOCS / "lab1_report_paper.pdf"
    shutil.copyfile(build / "lab1_report_paper.pdf", output)
    print(output)


if __name__ == "__main__":
    main()
