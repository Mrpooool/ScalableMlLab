# 论文版报告与复现

本次只调整报告呈现，保留所有实验测量与已执行 Notebook。
原版 `lab1_report.md/pdf` 保留供追溯；新版正式报告为
`lab1_report_paper.pdf`，可编辑版本为 `lab1_report_paper.docx`。

## 已采用的学术格式

- 使用 nature-paper2word 的 `en-nature` 预设，导出引擎为 `md-to-word-docx 3.6.0`。
- A4 纵向、2.54 cm 页边距；Word 正文 Times New Roman 12 pt、1.5 倍行距。
- 增加摘要与关键词，统一章节编号；保留 Task 1-6 和 Challenge 2 的对应标签。
- 12 张三线表、20 幅有编号题注的原始图；原有 71 行表格内容逐行保持一致。
- FGSM 为 Word 原生 OMML 公式，带公式编号与正文交叉引用。
- 正文采用 Nature CSL 编号引用，10 条参考文献由 BibTeX 自动生成。
- 论文版是课程实验报告的学术排版版本，不代表 Nature 期刊投稿合规认证。

## 重建报告

```bash
python -m pip install md-to-word-docx==3.6.0 pillow
md-to-word --version
python lab1/scripts/format_paper.py
md-to-word lab1/docs/lab1_report_paper.md --preset en-nature --lang en -o lab1/docs/lab1_report_paper.docx
python lab1/scripts/build_paper_pdf.py --pandoc /path/to/pandoc --tectonic /path/to/tectonic
```

`lab1_references.bib` 与 `nature.csl` 随源码提供。PDF 从同一规范化稿件编译，
不经过 Word 到 Markdown 的反向转换，避免图片与题注对应关系发生偏移。

## 校验范围

Word 导出通过引擎的语法预检、数值完整性检查和格式审计，并核查了 OOXML 中的
表格、图、原生公式、字体与参考文献。当前机器的 Word 后台 PDF 导出未能完成，
所以 PDF 使用 Pandoc/Tectonic 排版并单独检查页面。Word 与 PDF 的分页可能不同；
如需修改 Word 后直接提交，请先在 Word 中检查分页。

使用 `lab1_submission_paper.zip` 提交时，包内的 `lab1_report.pdf` 为论文新版，
Notebook 保留原始的 9 个已执行单元与 24 幅图。小组姓名、学号仍应按课程要求填写。
