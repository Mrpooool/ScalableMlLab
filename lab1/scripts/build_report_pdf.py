"""Render the unified Markdown report as a paginated PDF using ReportLab."""

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "lab1/docs/lab1_report.md"
OUTPUT = ROOT / "lab1/docs/lab1_report.pdf"
styles = getSampleStyleSheet()
styles["BodyText"].fontName = "Helvetica"
styles["BodyText"].fontSize = 10
styles["BodyText"].leading = 14
styles["BodyText"].spaceAfter = 7
styles["BodyText"].allowWidows = 0
styles["BodyText"].allowOrphans = 0
styles.add(ParagraphStyle("Caption", fontSize=9, leading=12, alignment=TA_CENTER,
                          textColor=colors.HexColor("#444444"), spaceAfter=13))
styles.add(ParagraphStyle("Cell", fontSize=8, leading=10, wordWrap="CJK"))
styles.add(ParagraphStyle("MathText", fontName="Courier", fontSize=9, leading=12,
                          leftIndent=12, spaceAfter=10))
WIDTH = A4[0] - 100


def inline(text):
    text = text.replace("—", "-").replace("–", "-").replace("×", "x").replace("→", "->")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#004791">\1</link>', text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor("#cccccc"))
    canvas.line(50, 40, A4[0] - 50, 40)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(50, 28, "Scalable Machine Learning | Lab 1 | Tasks 1-6 and Challenge 2")
    canvas.drawRightString(A4[0] - 50, 28, str(doc.page))


def main():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    flow = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            style = styles["Title" if level == 1 else "Heading2" if level == 2 else "Heading3"]
            flow.append(Paragraph(inline(line.lstrip("# ")), style))
            index += 1
        elif line.startswith("!["):
            caption, name = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line).groups()
            path = (SOURCE.parent / name).resolve()
            width, height = ImageReader(str(path)).getSize()
            ratio = min(WIDTH / width, 340 / height)
            flow.append(KeepTogether([Image(str(path), width=width * ratio, height=height * ratio),
                                      Paragraph(inline(caption), styles["Caption"])]))
            index += 1
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                values = [v.strip() for v in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[: -]+", value) for value in values):
                    rows.append(values)
                index += 1
            columns = len(rows[0])
            first = 1.65 if columns >= 5 else 1.25
            unit = WIDTH / (columns - 1 + first)
            widths = [first * unit] + [unit] * (columns - 1)
            cells = [[Paragraph(inline(value), styles["Cell"]) for value in row] for row in rows]
            table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf0f5")),
                ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.HexColor("#004791")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#999999")),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#999999")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            flow.extend([table, Spacer(1, 12)])
        elif line == r"\[":
            # The report contains one displayed FGSM formula; keep its meaning readable in PDF.
            while index < len(lines) and lines[index].strip() != r"\]":
                index += 1
            flow.append(Paragraph("x_adv = clip_[0,1](x + epsilon * sign(gradient_x L))", styles["MathText"]))
            index += 1
        else:
            paragraph = [line]
            index += 1
            if line.startswith("- "):
                flow.append(Paragraph(inline(line[2:]), styles["BodyText"], bulletText="-"))
                continue
            while index < len(lines) and lines[index].strip() and not lines[index].startswith(("#", "|", "![", "- ")):
                paragraph.append(lines[index].strip())
                index += 1
            flow.append(Paragraph(inline(" ".join(paragraph)), styles["BodyText"]))
    SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=50, leftMargin=50,
                      topMargin=45, bottomMargin=52,
                      title="Lab 1: Building and Evaluating Convolutional Neural Networks",
                      author="Lab 1 project team").build(flow, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
