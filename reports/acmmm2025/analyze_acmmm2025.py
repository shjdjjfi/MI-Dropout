#!/usr/bin/env python3
"""Analyze ACM MM 2025 accepted regular papers by research directions.

This script fetches the accepted regular papers page from the ACM MM 2025
WordPress API, extracts paper titles, applies a keyword-based direction
classifier, and writes CSV/Markdown/SVG outputs.
"""

import csv
import html
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

REGULAR_PAPERS_SLUG = "accepted-regular-papers"

# Priority order matters: the first matching direction is assigned.
DIRECTION_KEYWORDS = [
    (
        "Foundation Models / LLM / MLLM",
        [
            "llm", "mllm", "vlm", "large language model", "foundation model",
            "language model", "multimodal large", "gpt", "diffusion transformer",
            "instruction tuning", "prompt", "in-context", "agent"
        ],
    ),
    (
        "Generation / Diffusion / AIGC",
        [
            "diffusion", "generation", "generative", "text-to-image", "image-to-image",
            "video generation", "synthesis", "gan", "edit", "editing", "stylization",
            "inpainting", "outpainting", "3dgs", "gaussian splatting"
        ],
    ),
    (
        "Retrieval / Recommendation / Search",
        [
            "retrieval", "recommendation", "recommender", "search", "ranking",
            "matching", "query", "rerank", "cross-modal retrieval"
        ],
    ),
    (
        "Vision-Language / Multimodal Understanding",
        [
            "vision-language", "multimodal", "image caption", "visual question answering",
            "vqa", "cross-modal", "grounding", "alignment", "reasoning"
        ],
    ),
    (
        "Video Understanding / Temporal Modeling",
        [
            "video", "temporal", "action", "event", "trajectory", "tracking", "streaming"
        ],
    ),
    (
        "Audio / Speech / Music",
        [
            "audio", "speech", "music", "acoustic", "sound", "voice", "singing"
        ],
    ),
    (
        "3D / Embodied / Spatial",
        [
            "3d", "neural radiance", "nerf", "point cloud", "mesh", "embodied",
            "spatial", "scene", "depth"
        ],
    ),
    (
        "Medical / Bio / Healthcare",
        [
            "medical", "clinical", "health", "healthcare", "bio", "biomedical", "disease"
        ],
    ),
    (
        "Security / Privacy / Watermark",
        [
            "security", "privacy", "watermark", "backdoor", "adversarial", "robust",
            "defense", "attack", "forensics"
        ],
    ),
    (
        "Efficiency / Optimization / Compression",
        [
            "optimization", "zero-order", "zeroth-order", "efficient", "compression",
            "quantization", "pruning", "distillation", "acceleration", "low-rank"
        ],
    ),
    (
        "Graph / Knowledge / Causality",
        [
            "graph", "knowledge", "causal", "causality", "ontology", "knowledge graph"
        ],
    ),
]

OUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUT_DIR / "acmmm2025_regular_papers_by_direction.csv"
SVG_PATH = OUT_DIR / "acmmm2025_regular_papers_by_direction.svg"
MD_PATH = OUT_DIR / "acmmm2025_regular_papers_by_direction.md"


def fetch_page_content(slug: str) -> str:
    url = f"https://acmmm2025.org/wp-json/wp/v2/pages?slug={slug}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not payload:
        raise RuntimeError(f"Page not found for slug: {slug}")
    return html.unescape(payload[0]["content"]["rendered"])


def extract_titles(rendered_content: str) -> list[str]:
    # Format in source: <p>1 <b>Title</b><br /> Authors...</p>
    raw_titles = re.findall(r"<p>\s*\d+\s*<b>(.*?)</b>\s*<br\s*/?>", rendered_content, flags=re.I | re.S)
    clean = [re.sub(r"\s+", " ", re.sub(r"<.*?>", "", t)).strip() for t in raw_titles]
    return [t for t in clean if t]


def classify_title(title: str) -> str:
    t = title.lower()
    for direction, keywords in DIRECTION_KEYWORDS:
        if any(k in t for k in keywords):
            return direction
    return "Other / General Multimedia"


def build_svg(rows):
    width, height = 1300, 760
    margin_left, margin_right, margin_top, margin_bottom = 120, 50, 90, 220
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    max_count = max(r["count"] for r in rows)

    n = len(rows)
    bar_gap = 14
    bar_w = (chart_w - bar_gap * (n - 1)) / n

    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC", "#8CD17D", "#499894"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="650" y="42" text-anchor="middle" font-size="28" font-family="Arial" font-weight="bold">ACM MM 2025 Regular Papers by Research Direction</text>',
        '<text x="650" y="70" text-anchor="middle" font-size="16" font-family="Arial" fill="#444">Keyword-based classification from accepted regular paper titles</text>',
        f'<line x1="{margin_left}" y1="{margin_top+chart_h}" x2="{margin_left+chart_w}" y2="{margin_top+chart_h}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top+chart_h}" stroke="#222" stroke-width="2"/>'
    ]

    for i in range(0, 6):
        y = margin_top + chart_h - chart_h * i / 5
        v = int(max_count * i / 5)
        parts.append(f'<line x1="{margin_left-6}" y1="{y:.1f}" x2="{margin_left}" y2="{y:.1f}" stroke="#222"/>')
        parts.append(f'<text x="{margin_left-12}" y="{y+5:.1f}" text-anchor="end" font-size="12" font-family="Arial">{v}</text>')
        if i:
            parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left+chart_w}" y2="{y:.1f}" stroke="#ddd" stroke-dasharray="3,4"/>')

    for i, r in enumerate(rows):
        x = margin_left + i * (bar_w + bar_gap)
        h = (r["count"] / max_count) * chart_h
        y = margin_top + chart_h - h
        c = colors[i % len(colors)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{c}"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-size="12" font-family="Arial">{r["count"]}</text>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{margin_top+chart_h+20}" text-anchor="middle" font-size="11" font-family="Arial" transform="rotate(33 {x+bar_w/2:.1f} {margin_top+chart_h+20})">{r["direction"]}</text>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{margin_top+chart_h+46}" text-anchor="middle" font-size="10" fill="#555" font-family="Arial">{r["percentage"]:.2f}%</text>')

    parts.append('</svg>')
    return "\n".join(parts)


def main():
    content = fetch_page_content(REGULAR_PAPERS_SLUG)
    titles = extract_titles(content)

    counts = defaultdict(int)
    for t in titles:
        counts[classify_title(t)] += 1

    total = len(titles)
    rows = [
        {"direction": k, "count": v, "percentage": (v / total * 100 if total else 0)}
        for k, v in counts.items()
    ]
    rows.sort(key=lambda r: r["count"], reverse=True)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["direction", "count", "percentage"])
        for r in rows:
            w.writerow([r["direction"], r["count"], f"{r['percentage']:.4f}"])

    SVG_PATH.write_text(build_svg(rows), encoding="utf-8")

    lines = [
        "# ACM MM 2025 Regular Papers 方向分类统计（基于标题关键词）",
        "",
        "说明：这是**启发式关键词分类**，用于快速观察方向占比，不等同于官方 track 划分。",
        "",
        "| 方向 | 数量 | 占比 |",
        "|---|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['direction']} | {r['count']} | {r['percentage']:.2f}% |")
    lines += ["", f"Regular Papers 总计：**{total}**"]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Total regular papers extracted: {total}")
    print(CSV_PATH)
    print(SVG_PATH)
    print(MD_PATH)


if __name__ == "__main__":
    main()
