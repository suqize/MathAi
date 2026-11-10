"""Build Word and PDF grading reports with patent-style sections."""

from __future__ import annotations

from html import escape
from io import BytesIO
from typing import Any


STATE_LABELS = ("步骤正确", "计算错误", "定理误用", "逻辑跳跃", "条件误读", "步骤冗余", "步骤缺失")


def _probability_text(step: dict[str, Any], label: str) -> str:
    probabilities = step.get("status_probabilities") or {}
    value = probabilities.get(label)
    return f"{value:.1%}" if isinstance(value, (int, float)) else "未提供"


def _image_bytes(image: Any | None) -> bytes | None:
    if image is None:
        return None
    output = BytesIO()
    image.convert("RGB").save(output, format="PNG")
    return output.getvalue()


def build_word_report(question: str, result: dict[str, Any], image: Any | None = None,
                      heatmap: Any | None = None) -> bytes:
    """Create a DOCX report structured like the patent example report."""
    from docx import Document
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.72)
    section.right_margin = Inches(0.72)

    normal = document.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal.font.size = Pt(9.5)
    normal.font.color.rgb = RGBColor(43, 53, 49)
    for style_name in ("Title", "Heading 1", "Heading 2"):
        style = document.styles[style_name]
        style.font.name = "Microsoft YaHei"
        style.font.color.rgb = RGBColor(30, 85, 65)
    document.styles["Title"].font.size = Pt(21)
    document.styles["Heading 1"].font.size = Pt(15)
    document.styles["Heading 2"].font.size = Pt(11)

    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("数学解答过程步骤级批改报告")
    intro = document.add_paragraph("本报告按解答原始顺序汇总步骤状态、状态概率、自然语言反馈及图像区域关联。")
    intro.alignment = WD_ALIGN_PARAGRAPH.CENTER

    document.add_heading("第一部分  样例输入", level=1)
    document.add_heading("01  题目文本", level=2)
    document.add_paragraph(question)
    if image is not None:
        document.add_heading("02  原始几何题目图像", level=2)
        image_data = _image_bytes(image)
        if image_data:
            document.add_picture(BytesIO(image_data), width=Inches(5.8))
    document.add_heading("03  学生解答步骤文本序列", level=2)
    for step in result.get("steps", []):
        document.add_paragraph(f"S{step['index']}  {step['text']}")

    document.add_heading("第二部分  步骤级批改输出", level=1)
    for step in result.get("steps", []):
        document.add_heading(f"步骤 S{step['index']}  {step.get('status_detail', step.get('status', '存疑'))}", level=2)
        document.add_paragraph(f"解答内容：{step['text']}")
        document.add_paragraph(f"状态概率置信度：{step.get('confidence', 0):.1%}")
        table = document.add_table(rows=1, cols=2)
        table.style = "Table Grid"
        table.rows[0].cells[0].text = "七类状态"
        table.rows[0].cells[1].text = "概率"
        for label in STATE_LABELS:
            cells = table.add_row().cells
            cells[0].text = label
            cells[1].text = _probability_text(step, label)
        for row in table.rows:
            for cell in row.cells:
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        document.add_paragraph(f"批改反馈：{step.get('feedback', '')}")
        reasoning = step.get("reasoning", "")
        if reasoning:
            document.add_paragraph(f"判定依据：{reasoning}")

    document.add_heading("第三部分  错误区域可视化标注", level=1)
    if heatmap is not None:
        document.add_paragraph("热力图显示步骤与题图区域的关联强度，颜色由冷到暖表示响应逐渐增强。")
        heatmap_data = _image_bytes(heatmap)
        if heatmap_data:
            document.add_picture(BytesIO(heatmap_data), width=Inches(5.8))
    else:
        document.add_paragraph("本次结果未提供可用的图像区域热力图。")

    document.add_heading("综合批改反馈", level=1)
    document.add_paragraph(result.get("summary", "批改完成。"))
    links = result.get("logic_links", [])
    if links:
        document.add_heading("步骤逻辑关系", level=2)
        for link in links:
            document.add_paragraph(
                f"步骤 {link.get('from', '?')} 到步骤 {link.get('to', '?')}："
                f"{link.get('relation', '推导')}；{link.get('label', '')}",
                style="List Bullet",
            )
    notes = result.get("review_notes", [])
    if notes:
        document.add_heading("复核提醒", level=2)
        for note in notes:
            document.add_paragraph(str(note), style="List Bullet")

    output = BytesIO()
    document.save(output)
    return output.getvalue()


def build_pdf_report(question: str, result: dict[str, Any], image: Any | None = None,
                     heatmap: Any | None = None) -> bytes:
    """Create a CJK-capable PDF report with patent-style sections."""
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontName="STSong-Light",
                                 fontSize=19, leading=26, textColor=colors.HexColor("#215c46"),
                                 alignment=TA_CENTER, spaceAfter=5 * mm)
    body = ParagraphStyle("ReportBody", parent=styles["BodyText"], fontName="STSong-Light",
                          fontSize=9, leading=14, textColor=colors.HexColor("#28332e"), spaceAfter=2.5 * mm)
    heading = ParagraphStyle("ReportHeading", parent=styles["Heading1"], fontName="STSong-Light",
                             fontSize=14, leading=19, textColor=colors.HexColor("#215c46"),
                             spaceBefore=5 * mm, spaceAfter=3 * mm)
    subheading = ParagraphStyle("ReportSubheading", parent=styles["Heading2"], fontName="STSong-Light",
                                fontSize=10.5, leading=15, textColor=colors.HexColor("#215c46"),
                                spaceBefore=3 * mm, spaceAfter=2 * mm)
    small = ParagraphStyle("ReportSmall", parent=body, fontSize=7.5, leading=10, spaceAfter=0)

    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
                            topMargin=17 * mm, bottomMargin=17 * mm, title="数学解答过程步骤级批改报告")
    story: list[Any] = [
        Paragraph("数学解答过程步骤级批改报告", title_style),
        Paragraph("本报告按解答原始顺序汇总步骤状态、状态概率、自然语言反馈及图像区域关联。", body),
        Paragraph("第一部分  样例输入", heading),
        Paragraph("01  题目文本", subheading),
        Paragraph(escape(question).replace("\n", "<br/>"), body),
    ]

    def add_image(target: Any | None, width: float = 165 * mm) -> None:
        image_data = _image_bytes(target)
        if not image_data:
            return
        with PILImage.open(BytesIO(image_data)) as source:
            iw, ih = source.size
        rendered_width = min(width, iw * mm / 96)
        rendered_height = rendered_width * ih / iw
        story.append(Image(BytesIO(image_data), width=rendered_width, height=rendered_height))
        story.append(Spacer(1, 3 * mm))

    if image is not None:
        story.append(Paragraph("02  原始几何题目图像", subheading))
        add_image(image)
    story.append(Paragraph("03  学生解答步骤文本序列", subheading))
    for step in result.get("steps", []):
        story.append(Paragraph(f"S{step['index']}  {escape(str(step['text']))}", body))

    story.append(Paragraph("第二部分  步骤级批改输出", heading))
    for step in result.get("steps", []):
        story.append(Paragraph(
            f"步骤 S{step['index']}  {escape(str(step.get('status_detail', step.get('status', '存疑'))))}",
            subheading,
        ))
        story.append(Paragraph(f"解答内容：{escape(str(step['text']))}", body))
        story.append(Paragraph(f"状态概率置信度：{float(step.get('confidence', 0)):.1%}", body))
        probability_data = [[Paragraph("状态类别", small), Paragraph("概率", small)]]
        for label in STATE_LABELS:
            probability_data.append([Paragraph(label, small), Paragraph(_probability_text(step, label), small)])
        probability_table = Table(probability_data, colWidths=[90 * mm, 35 * mm], repeatRows=1)
        probability_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f3ed")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#215c46")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#dce4de")),
            ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.extend([probability_table, Spacer(1, 2 * mm)])
        story.append(Paragraph(f"批改反馈：{escape(str(step.get('feedback', '')))}", body))
        if step.get("reasoning"):
            story.append(Paragraph(f"判定依据：{escape(str(step['reasoning']))}", body))

    story.append(Paragraph("第三部分  错误区域可视化标注", heading))
    if heatmap is not None:
        story.append(Paragraph("热力图中颜色由冷到暖表示响应逐渐增强。", body))
        add_image(heatmap)
    else:
        story.append(Paragraph("本次结果未提供可用的图像区域热力图。", body))

    story.append(Paragraph("综合批改反馈", heading))
    story.append(Paragraph(escape(str(result.get("summary", "批改完成。"))), body))
    for link in result.get("logic_links", []):
        story.append(Paragraph(
            f"步骤 {escape(str(link.get('from', '?')))} 到步骤 {escape(str(link.get('to', '?')))}："
            f"{escape(str(link.get('relation', '推导')))}；{escape(str(link.get('label', '')))}", body
        ))
    for note in result.get("review_notes", []):
        story.append(Paragraph(f"复核提醒：{escape(str(note))}", body))

    def draw_page_number(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setFont("STSong-Light", 8)
        canvas.setFillColor(colors.HexColor("#76817b"))
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"第 {document.page} 页")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_page_number, onLaterPages=draw_page_number)
    return output.getvalue()
