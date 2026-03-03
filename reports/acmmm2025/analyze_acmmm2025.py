#!/usr/bin/env python3
import csv
import html
import json
import re
import urllib.request
from pathlib import Path

PAGES = {
    "Regular Papers": "accepted-regular-papers",
    "Datasets": "accepted-papers-datasets",
    "Brave New Ideas": "accepted-papers-brave-new-ideas",
    "Demo/Video": "accepted-papers-demo-video",
    "Open Source Software": "accepted-papers-open-source-software",
    "Doctoral Symposium": "accepted-papers-doctoral-symposium",
    "Interactive Art": "accepted-papers-interactive-art",
}

OUT_DIR = Path(__file__).resolve().parent
CSV_CATEGORY_PATH = OUT_DIR / "acmmm2025_accepted_papers_by_category.csv"
CSV_ALL_PATH = OUT_DIR / "acmmm2025_all_papers_screening.csv"
MD_PATH = OUT_DIR / "acmmm2025_accepted_papers_summary.md"

EASY_INFERENCE_KEYWORDS = [
    "normalization", "calibration", "re-ranking", "reranking", "sinkhorn", "hubness",
    "score", "logit", "temperature",
]

EASY_TRAINING_KEYWORDS = [
    "distillation", "kd", "weighted", "reweight", "regularization", "contrastive",
    "margin", "debias", "bias", "hard negative",
]

PLUGIN_KEYWORDS = ["prompt", "prompting", "adapter", "lora", "token", "memory", "bank", "plug-and-play"]

EASY_STYLE_KEYWORDS = ["simple", "efficient", "lightweight", "plug-and-play"]

PASS_KEYWORDS = [
    "unified framework", "end-to-end system", "multi-stage", "tri-branch", "hierarchical transformer",
    "new dataset", "benchmark", "annotation",
]

TASK_KEYWORDS = {
    "Retrieval": ["retrieval", "search", "rerank"],
    "Video": ["video", "temporal", "moment", "action"],
    "VLM": ["vlm", "multimodal", "vision-language", "vision language", "mllm", "llm"],
    "Seg": ["segmentation", "segment", "mask"],
    "Audio": ["audio", "speech", "music"],
}


def fetch_page_content(slug: str) -> str:
    url = f"https://acmmm2025.org/wp-json/wp/v2/pages?slug={slug}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not payload:
        raise RuntimeError(f"Page not found for slug: {slug}")
    return html.unescape(payload[0]["content"]["rendered"])


def clean_html(text: str) -> str:
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_entries(rendered_content: str):
    pattern = re.compile(r"<p>\s*(\d+)\s*<b>(.*?)</b><br\s*/?>\s*(.*?)</p>", re.IGNORECASE)
    entries = []
    for num, title_html, authors_html in pattern.findall(rendered_content):
        entries.append(
            {
                "id": int(num),
                "title": clean_html(title_html),
                "authors": clean_html(authors_html),
            }
        )
    return entries


def contains_any(text: str, keywords):
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def hit_keywords(text: str, keywords):
    lowered = text.lower()
    return [k for k in keywords if k in lowered]


def classify_task(title: str) -> str:
    lowered = title.lower()
    for task, kws in TASK_KEYWORDS.items():
        if any(k in lowered for k in kws):
            return task
    return "Other"


def classify_type(title: str) -> str:
    lowered = title.lower()
    if contains_any(lowered, EASY_INFERENCE_KEYWORDS):
        return "Inference postproc"
    if contains_any(lowered, EASY_TRAINING_KEYWORDS):
        return "Loss / Training trick"
    if contains_any(lowered, PLUGIN_KEYWORDS):
        return "Adapter-Prompt"
    if contains_any(lowered, ["data-free"]):
        return "Data-free"
    return "Other"


def score_entry(title: str):
    lowered = title.lower()

    s1 = 1 if (
        contains_any(lowered, EASY_INFERENCE_KEYWORDS)
        or contains_any(lowered, EASY_TRAINING_KEYWORDS)
        or contains_any(lowered, PLUGIN_KEYWORDS)
    ) else 0
    s2 = 0 if contains_any(lowered, ["dataset", "benchmark", "annotation"]) else 1
    s3 = 0  # title-only pass, no code lookup
    s4 = 1 if (
        contains_any(lowered, EASY_INFERENCE_KEYWORDS)
        or contains_any(lowered, EASY_TRAINING_KEYWORDS)
        or contains_any(lowered, PLUGIN_KEYWORDS)
    ) else 0

    total = s1 + s2 + s3 + s4

    if contains_any(lowered, PASS_KEYWORDS):
        verdict = "PASS"
    elif contains_any(lowered, EASY_INFERENCE_KEYWORDS + EASY_TRAINING_KEYWORDS + PLUGIN_KEYWORDS + EASY_STYLE_KEYWORDS):
        verdict = "KEEP"
    else:
        verdict = "MAYBE"

    ease = 1 + s1 + s2 + s4
    ease = min(5, ease)

    return s1, s2, s3, s4, total, verdict, ease


def main():
    category_rows = []
    all_rows = []

    for category, slug in PAGES.items():
        content = fetch_page_content(slug)
        entries = extract_entries(content)
        category_rows.append({"category": category, "count": len(entries)})

        for e in entries:
            s1, s2, s3, s4, total, verdict, ease = score_entry(e["title"])
            all_rows.append(
                {
                    "category": category,
                    "paper_id": e["id"],
                    "title": e["title"],
                    "authors": e["authors"],
                    "task": classify_task(e["title"]),
                    "type": classify_type(e["title"]),
                    "needs_training": "N" if classify_type(e["title"]) == "Inference postproc" else "Y",
                    "extra_data": "Y" if contains_any(e["title"].lower(), ["dataset", "benchmark", "annotation"]) else "N",
                    "code": "Unknown",
                    "s1_mod_pos": s1,
                    "s2_no_extra_data": s2,
                    "s3_has_code": s3,
                    "s4_pluggable": s4,
                    "score_total": total,
                    "verdict": verdict,
                    "ease": ease,
                    "keyword_hits": ", ".join(hit_keywords(e["title"], EASY_INFERENCE_KEYWORDS + EASY_TRAINING_KEYWORDS + PLUGIN_KEYWORDS + EASY_STYLE_KEYWORDS)),
                }
            )

    category_rows.sort(key=lambda x: x["count"], reverse=True)
    total_count = sum(r["count"] for r in category_rows)
    for r in category_rows:
        r["percentage"] = r["count"] / total_count * 100 if total_count else 0.0

    with CSV_CATEGORY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "count", "percentage"])
        for r in category_rows:
            writer.writerow([r["category"], r["count"], f"{r['percentage']:.4f}"])

    all_rows.sort(key=lambda x: (x["category"], x["paper_id"]))
    with CSV_ALL_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    keep_rows = [r for r in all_rows if r["verdict"] == "KEEP"]
    maybe_rows = [r for r in all_rows if r["verdict"] == "MAYBE"]
    pass_rows = [r for r in all_rows if r["verdict"] == "PASS"]

    lines = [
        "# ACM MM 2025 全量 accepted papers 快速筛选（标题信号版）",
        "",
        "数据来源：ACM MM 2025 官网所有 accepted 页面（WordPress JSON API）。",
        "",
        "## 1) 全量覆盖统计",
        "",
        "| 类别 | 数量 | 占比 |",
        "|---|---:|---:|",
    ]
    for r in category_rows:
        lines.append(f"| {r['category']} | {r['count']} | {r['percentage']:.2f}% |")

    lines.extend([
        "",
        f"总计：**{total_count}** 篇（已遍历全部 accepted 列表条目）。",
        "",
        "## 2) 按标题规则的可实现性初筛",
        "",
        f"* KEEP（倾向小改动/易落地）：**{len(keep_rows)}**",
        f"* MAYBE（信息不足，需二次确认）：**{len(maybe_rows)}**",
        f"* PASS（标题即显示实现成本较高）：**{len(pass_rows)}**",
        "",
        "说明：本次为**全量标题级别**筛选；S3（是否有代码）默认 Unknown/0，后续可对 KEEP 集合再批量补 arXiv/GitHub 校验。",
        "",
        "## 3) 输出文件",
        "",
        "* `acmmm2025_all_papers_screening.csv`：所有论文逐条记录（含 task/type/score/verdict/ease）。",
        "* `acmmm2025_accepted_papers_by_category.csv`：按类别统计。",
    ])

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Total papers: {total_count}")
    print(f"KEEP={len(keep_rows)} MAYBE={len(maybe_rows)} PASS={len(pass_rows)}")
    print(f"Wrote {CSV_CATEGORY_PATH}")
    print(f"Wrote {CSV_ALL_PATH}")
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()
