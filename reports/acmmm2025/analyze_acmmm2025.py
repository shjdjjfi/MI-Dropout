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
CSV_PATH = OUT_DIR / "acmmm2025_accepted_papers_by_category.csv"
SVG_PATH = OUT_DIR / "acmmm2025_accepted_papers_by_category.svg"
MD_PATH = OUT_DIR / "acmmm2025_accepted_papers_summary.md"


def fetch_page_content(slug: str) -> str:
    url = f"https://acmmm2025.org/wp-json/wp/v2/pages?slug={slug}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not payload:
        raise RuntimeError(f"Page not found for slug: {slug}")
    return html.unescape(payload[0]["content"]["rendered"])


def count_accepted_entries(rendered_content: str) -> int:
    return len(re.findall(r"<p>\s*\d+\s*<b>", rendered_content))


def build_svg(rows):
    width, height = 1200, 720
    margin_left, margin_right, margin_top, margin_bottom = 110, 40, 80, 170
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    max_count = max(r["count"] for r in rows)
    bar_gap = 20
    n = len(rows)
    bar_w = (chart_w - bar_gap * (n - 1)) / n

    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B", "#B279A2"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="600" y="40" text-anchor="middle" font-size="28" font-family="Arial" font-weight="bold">ACM MM 2025 Accepted Papers by Category</text>',
        '<text x="600" y="66" text-anchor="middle" font-size="16" font-family="Arial" fill="#444">Source: acmmm2025.org accepted papers pages</text>',
        f'<line x1="{margin_left}" y1="{margin_top + chart_h}" x2="{margin_left + chart_w}" y2="{margin_top + chart_h}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_h}" stroke="#222" stroke-width="2"/>',
    ]

    ticks = 5
    for i in range(ticks + 1):
        y = margin_top + chart_h - (chart_h * i / ticks)
        val = int(max_count * i / ticks)
        parts.append(f'<line x1="{margin_left-6}" y1="{y:.1f}" x2="{margin_left}" y2="{y:.1f}" stroke="#222"/>')
        parts.append(f'<text x="{margin_left-12}" y="{y+5:.1f}" text-anchor="end" font-size="12" font-family="Arial">{val}</text>')
        if i > 0:
            parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left+chart_w}" y2="{y:.1f}" stroke="#ddd" stroke-dasharray="3,4"/>')

    for i, row in enumerate(rows):
        x = margin_left + i * (bar_w + bar_gap)
        h = chart_h * row["count"] / max_count
        y = margin_top + chart_h - h
        color = colors[i % len(colors)]

        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-size="12" font-family="Arial">{row["count"]}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{margin_top + chart_h + 20}" text-anchor="middle" font-size="12" font-family="Arial" transform="rotate(35 {x + bar_w/2:.1f} {margin_top + chart_h + 20})">{row["category"]}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{margin_top + chart_h + 45}" text-anchor="middle" font-size="11" fill="#555" font-family="Arial">{row["percentage"]:.2f}%</text>')

    parts.append('</svg>')
    return "\n".join(parts)


def main():
    rows = []
    for category, slug in PAGES.items():
        content = fetch_page_content(slug)
        rows.append({"category": category, "count": count_accepted_entries(content)})

    rows.sort(key=lambda x: x["count"], reverse=True)
    total = sum(r["count"] for r in rows)
    for r in rows:
        r["percentage"] = (r["count"] / total * 100) if total else 0

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "count", "percentage"])
        for r in rows:
            writer.writerow([r["category"], r["count"], f"{r['percentage']:.4f}"])

    SVG_PATH.write_text(build_svg(rows), encoding="utf-8")

    lines = [
        "# ACM MM 2025 Accepted Papers 分类统计",
        "",
        "数据来源：ACM MM 2025 官网各 accepted 页面（通过 WordPress JSON API 抓取并统计每条编号论文）。",
        "",
        "| 类别 | 数量 | 占比 |",
        "|---|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['category']} | {r['count']} | {r['percentage']:.2f}% |")
    lines.append("")
    lines.append(f"总计：**{total}**")
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:")
    print(CSV_PATH)
    print(SVG_PATH)
    print(MD_PATH)


if __name__ == "__main__":
    main()
