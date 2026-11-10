"""Streamlit user interface."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from src.config import ConfigError, get_active_provider
from src.patent_example import load_patent_example

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;600;700;800&display=swap');
:root{--ink:#17221f;--muted:#76817b;--line:#e8ece7;--green:#215c46;--mint:#e9f3ed;--canvas:#f7f8f5;}
html,body,[class*="css"]{font-family:'DM Sans','Noto Sans SC',sans-serif;color:var(--ink)}
.stApp{background:var(--canvas)} .block-container{max-width:1460px;padding:2.1rem 3rem 4rem}
header[data-testid="stHeader"]{background:transparent} #MainMenu,footer{visibility:hidden}
.hero{display:flex;justify-content:space-between;align-items:flex-end;padding:1.25rem 0 1.65rem;border-bottom:1px solid var(--line);margin-bottom:1.7rem}
.eyebrow{font-size:.72rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--green)}
.hero h1{font-size:2.05rem;letter-spacing:-.04em;margin:.4rem 0 .3rem;font-weight:700}.hero p{color:var(--muted);margin:0;font-size:.93rem}
.badge{border:1px solid #dce6df;background:white;color:var(--green);border-radius:99px;padding:.55rem .85rem;font-size:.76rem;font-weight:600}
.section-title{font-size:.9rem;font-weight:700;margin:0 0 .75rem;display:flex;align-items:center;gap:.5rem}
.panel{background:white;border:1px solid var(--line);border-radius:16px;padding:1rem 1.15rem;margin-bottom:.9rem}
.metric{background:white;border:1px solid var(--line);border-radius:14px;padding:1rem 1.1rem}.metric-label{font-size:.76rem;color:var(--muted)}.metric-value{font-size:1.45rem;font-weight:700;margin-top:.35rem}
.step-card{background:white;border:1px solid var(--line);border-radius:14px;padding:1rem 1.15rem;margin:.6rem 0}
.step-head{display:flex;justify-content:space-between;gap:.6rem;align-items:center}.step-label{font-weight:700;font-size:.87rem}.status{font-size:.72rem;font-weight:700;border-radius:99px;padding:.3rem .62rem}
.correct{background:#e7f4ec;color:#27714b}.questionable{background:#fff3da;color:#986814}.incorrect{background:#fde9e7;color:#a9443c}.unsupported{background:#edf0f2;color:#5c6870}
.step-body{font-size:.9rem;line-height:1.65;margin:.6rem 0;color:#27322e}.feedback{font-size:.83rem;color:#56645d;line-height:1.65}.confidence{font-size:.74rem;color:var(--muted);margin-top:.45rem}
.note{background:#f0f5f1;border-left:3px solid #78a58c;padding:.75rem .9rem;border-radius:0 8px 8px 0;color:#52625a;font-size:.78rem;line-height:1.6}
div.stButton>button{border-radius:10px;font-weight:600;min-height:2.7rem} div.stButton>button[kind="primary"]{background:var(--green);border-color:var(--green)}
.stTextArea textarea,.stTextInput input{border-radius:10px}.stFileUploader section{border-radius:12px}
</style>
"""

STATUS_TEXT = {"correct": "正确", "questionable": "存疑", "incorrect": "错误", "unsupported": "缺少依据"}


def _overlay(image: Any, steps: list[dict[str, Any]]) -> Any:
    """Blend a smooth, diffuse heatmap derived from the model's region predictions."""
    import numpy as np
    from PIL import Image, ImageFilter

    canvas = image.convert("RGB").copy()
    w, h = canvas.size
    grid_w = min(320, max(64, w // 4))
    grid_h = min(320, max(64, h // 4))
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, grid_w), np.linspace(0, 1, grid_h))
    attention = np.zeros((grid_h, grid_w), dtype=np.float32)
    for step in steps:
        region = step.get("region")
        if not region or step["status"] == "correct":
            continue
        center_x = float(region["x"]) + float(region["width"]) / 2
        center_y = float(region["y"]) + float(region["height"]) / 2
        # Spread attention beyond the predicted box so the result reads as a heatmap,
        # rather than a highlighted rectangle.
        sigma_x = max(float(region["width"]) * 1.25, 0.055)
        sigma_y = max(float(region["height"]) * 1.5, 0.065)
        gaussian = np.exp(-0.5 * (((x_grid - center_x) / sigma_x) ** 2
                                  + ((y_grid - center_y) / sigma_y) ** 2))
        attention = np.maximum(attention, gaussian.astype(np.float32))

    if not np.any(attention):
        return canvas

    # A soft blue → cyan → yellow → orange → red palette like the patent example.
    stops = np.array([0.0, 0.18, 0.38, 0.58, 0.76, 0.90, 1.0], dtype=np.float32)
    colors = np.array([
        [80, 145, 245], [95, 210, 245], [130, 238, 200], [250, 245, 125],
        [255, 175, 80], [245, 95, 68], [220, 45, 65],
    ], dtype=np.float32)
    heat_rgb = np.stack([
        np.interp(attention, stops, colors[:, channel]) for channel in range(3)
    ], axis=-1).astype(np.uint8)
    heat_alpha = np.clip(attention * 0.62, 0, 0.58)

    heat_image = Image.fromarray(heat_rgb, mode="RGB").resize(
        (w, h), Image.Resampling.BICUBIC
    )
    alpha_image = Image.fromarray((heat_alpha * 255).astype(np.uint8), mode="L").resize(
        (w, h), Image.Resampling.BICUBIC
    )
    alpha_image = alpha_image.filter(ImageFilter.GaussianBlur(radius=max(2, min(w, h) * 0.012)))
    return Image.composite(heat_image, canvas, alpha_image)


def _show_results(result: dict[str, Any], image: Any | None) -> None:
    steps = result["steps"]
    avg_conf = sum(s["confidence"] for s in steps) / max(1, len(steps))
    good = sum(s["status"] == "correct" for s in steps)
    c1, c2, c3 = st.columns(3)
    for col, label, value in ((c1, "步骤数量", str(len(steps))), (c2, "判定正确", f"{good} / {len(steps)}"), (c3, "平均置信度", f"{avg_conf:.0%}")):
        col.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">✳ 总体评价</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="panel">{escape(result.get("summary", "批改完成。"))}</div>', unsafe_allow_html=True)
    st.markdown("**七类步骤状态概率**")
    import pandas as pd
    probability_rows = []
    missing_probabilities = False
    for step in steps:
        probabilities = step.get("status_probabilities")
        if not probabilities:
            missing_probabilities = True
            probabilities = {}
        row = {"步骤": f"S{step['index']}", "判定": step.get("status_detail", STATUS_TEXT.get(step["status"], "存疑"))}
        for label in ("步骤正确", "计算错误", "定理误用", "逻辑跳跃", "条件误读", "步骤冗余", "步骤缺失"):
            value = probabilities.get(label)
            row[label] = f"{value:.1%}" if value is not None else "—"
        probability_rows.append(row)
    st.dataframe(pd.DataFrame(probability_rows), hide_index=True, use_container_width=True)
    if missing_probabilities:
        st.caption("部分步骤未返回完整的七类概率分布；表格以“—”标出，未使用估算值补齐。")
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown('<div class="section-title">▤ 步骤分析</div>', unsafe_allow_html=True)
        for step in steps:
            status = step["status"]
            title = step.get("status_detail", STATUS_TEXT.get(status, "存疑"))
            body = f'''<div class="step-card"><div class="step-head"><span class="step-label">步骤 {step['index']} · {escape(step['logic_role'] or '解答步骤')}</span><span class="status {status}">{title}</span></div><div class="step-body">{escape(step['text'])}</div><div class="feedback">{escape(step['feedback'])}</div><div class="confidence">置信度 {step['confidence']:.0%}　{escape(step['reasoning'])}</div></div>'''
            st.markdown(body, unsafe_allow_html=True)
        links = result.get("logic_links", [])
        if links:
            st.markdown('<div class="section-title">↗ 步骤逻辑关系</div>', unsafe_allow_html=True)
            for link in links:
                st.markdown(f"- 步骤 {link.get('from','?')} → 步骤 {link.get('to','?')}：{link.get('relation','推导')} · {link.get('label','')}")
    with right:
        st.markdown('<div class="section-title">⌖ 图文区域关联</div>', unsafe_allow_html=True)
        if image:
            st.image(_overlay(image, steps), use_container_width=True)
        else:
            st.markdown('<div class="panel">未上传题目图片，当前仅依据文字批改。</div>', unsafe_allow_html=True)
        notes = result.get("review_notes", [])
        if notes:
            st.markdown('<div class="section-title">⚑ 复核提醒</div>', unsafe_allow_html=True)
            for note in notes:
                st.markdown(f'<div class="note">{escape(str(note))}</div>', unsafe_allow_html=True)
    st.download_button("下载批改报告 JSON", json.dumps(result, ensure_ascii=False, indent=2),
                       file_name="math_grading_report.json", mime="application/json")


def _show_patent_example(sample: dict[str, Any]) -> None:
    """Render the input/output example transcribed from the patent DOCX."""
    expected = sample["expected_output"]
    st.markdown('<div class="section-title">文档中的预计输出 · 模拟示例</div>', unsafe_allow_html=True)
    st.caption("以下为系统预设案例的参考结果，用于展示步骤批改报告格式，不代表当前输入的评估结论。")
    statuses = [step["status"] for step in expected["steps"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("批改步骤", len(statuses))
    c2.metric("步骤正确", f"{statuses.count('步骤正确')} / {len(statuses)}")
    c3.metric("条件误读", statuses.count("条件误读"))
    st.markdown(f'<div class="panel">{escape(expected["summary"])}</div>', unsafe_allow_html=True)

    import pandas as pd
    probability_rows = []
    for item in expected["steps"]:
        row = {"步骤": f"S{item['index']}", "预计状态": item["status"], "置信度": f"{item['confidence']:.0%}"}
        row.update({label: f"{probability:.0%}" for label, probability in item["probabilities"].items()})
        probability_rows.append(row)
    st.markdown("**七类步骤状态概率（文档模拟值）**")
    st.dataframe(pd.DataFrame(probability_rows), hide_index=True, use_container_width=True)

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("**逐步骤反馈**")
        for item in expected["steps"]:
            css_status = "incorrect" if item["status"] == "条件误读" else "correct"
            text = sample["input"]["steps"][item["index"] - 1]
            st.markdown(
                f'''<div class="step-card"><div class="step-head"><span class="step-label">S{item['index']}</span><span class="status {css_status}">{escape(item['status'])} · {item['confidence']:.0%}</span></div><div class="step-body">{escape(text)}</div><div class="feedback">{escape(item['feedback'])}</div></div>''',
                unsafe_allow_html=True,
            )
    with right:
        st.markdown("**原始题图 / S3 区域热力图**")
        st.image(sample["input"]["image"], caption="原始几何题目图像", use_container_width=True)
        st.image(sample["input"]["heatmap_image"], caption="S3 错误区域注意力热力图（文档模拟）", use_container_width=True)

    report = {
        "source_document": sample["source_document"],
        "question": sample["input"]["question"],
        "steps": [{**item, "text": sample["input"]["steps"][item["index"] - 1]} for item in expected["steps"]],
        "summary": expected["summary"],
    }
    st.download_button("下载文档样例报告 JSON", json.dumps(report, ensure_ascii=False, indent=2),
                       file_name="pgps9k_patent_example_report.json", mime="application/json")


def render_page(grader: Callable[..., dict[str, Any]], error_type: type[Exception]) -> None:
    sample = load_patent_example()
    sample_input = sample["input"]
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('''<div class="hero"><div><div class="eyebrow">MathVision · AI Geometry Review</div><h1>几何解答步骤批改</h1><p>结合题目图像与推理过程，逐步检查解答并生成可解释反馈。</p></div><div class="badge">● MathVision 智能评估模型</div></div>''', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### MathVision 智能评估")
        try:
            _, provider = get_active_provider()
            model = provider["model"]
            st.caption("逐步分析几何推理、条件使用与图文一致性。")
        except ConfigError:
            model = MODEL
            st.caption("智能评估服务暂不可用，请联系系统维护人员。")
        st.markdown("---")
        st.caption("提供步骤判定、反馈建议与图形关联分析。")
    col_input, col_result = st.columns([.9, 1.1], gap="large")
    with col_input:
        st.markdown('<div class="section-title">01　题目与作答</div>', unsafe_allow_html=True)
        st.caption(f"默认输入来源：{sample['source_document']}")
        question = st.text_area("题目描述", value=sample_input["question"], height=115)
        uploaded = st.file_uploader("上传题目图片（可选）", type=["png", "jpg", "jpeg", "webp"])
        image = None
        if uploaded:
            from PIL import Image
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption=f"题目图片 · {image.width} × {image.height}", use_container_width=True)
        else:
            from PIL import Image
            image = Image.open(sample_input["image"]).convert("RGB")
            st.image(image, caption="文档中的原始几何题图", use_container_width=True)
        raw_steps = st.text_area("学生解答步骤（每行一步）", value="\n".join(sample_input["steps"]), height=250)
        steps = [line.strip().lstrip("0123456789.、) ") for line in raw_steps.splitlines() if line.strip()]
        st.caption(f"已识别 {len(steps)} 个解答步骤")
        run = st.button("✦ 开始智能评估", type="primary", use_container_width=True)
        st.markdown('<div class="note">模型给出的错误区域是视觉定位建议。几何图形理解可能存在偏差，低置信结果请人工复核。</div>', unsafe_allow_html=True)
    with col_result:
        st.markdown('<div class="section-title">02　批改结果</div>', unsafe_allow_html=True)
        if run:
            if not question.strip() or not steps:
                st.error("请填写题目并至少提供一个解答步骤。")
            else:
                with st.spinner("正在分析题目图像与步骤逻辑…"):
                    try:
                        image_bytes = uploaded.getvalue() if uploaded else Path(sample_input["image"]).read_bytes()
                        result = grader(question, steps, image_bytes=image_bytes,
                                        image_mime=uploaded.type if uploaded else "image/png", model=model.strip())
                        st.session_state["grading_result"] = result
                        st.session_state["grading_image"] = image
                    except error_type:
                        st.error("暂时无法完成评估，请稍后重试。")
                        st.caption("如果问题持续，请联系系统维护人员。")
        if "grading_result" in st.session_state:
            st.markdown("**MathVision 智能评估结果**")
            _show_results(st.session_state["grading_result"], st.session_state.get("grading_image"))
        else:
            _show_patent_example(sample)
