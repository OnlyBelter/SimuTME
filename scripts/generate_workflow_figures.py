from pathlib import Path

def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _wrap_text(text: str, max_len: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur and cur_len + add > max_len:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        lines.append(" ".join(cur))
    return lines


def _svg_flow(title: str, steps: list[tuple[str, str]], width: int, height: int) -> str:
    margin_x = 90
    title_y = 70
    box_x = margin_x
    box_w = width - 2 * margin_x
    box_h = 95
    gap = 34
    start_y = 140

    svg: list[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append("<defs>")
    svg.append(
        '<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#2f2f2f"/></marker>'
    )
    svg.append("</defs>")

    svg.append(f'<text x="{width/2}" y="{title_y}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="28" fill="#111">{_escape(title)}</text>')

    boxes: list[tuple[int, int, int, int]] = []
    for i, (label, color) in enumerate(steps):
        y = start_y + i * (box_h + gap)
        boxes.append((box_x, y, box_w, box_h))
        svg.append(f'<rect x="{box_x}" y="{y}" width="{box_w}" height="{box_h}" rx="12" ry="12" fill="{color}" stroke="#2f2f2f" stroke-width="2"/>')

        lines = _wrap_text(label, 52)
        line_h = 20
        text_block_h = line_h * len(lines)
        text_y0 = y + (box_h - text_block_h) / 2 + line_h - 4
        for j, line in enumerate(lines):
            svg.append(
                f'<text x="{box_x + box_w/2}" y="{text_y0 + j*line_h}" text-anchor="middle" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="18" fill="#111">{_escape(line)}</text>'
            )

    for i in range(len(boxes) - 1):
        x, y, w, h = boxes[i]
        x2, y2, w2, h2 = boxes[i + 1]
        sx = x + w / 2
        sy = y + h
        ex = x2 + w2 / 2
        ey = y2
        svg.append(f'<line x1="{sx}" y1="{sy}" x2="{ex}" y2="{ey}" stroke="#2f2f2f" stroke-width="2" marker-end="url(#arrow)"/>')

    svg.append("</svg>")
    return "\n".join(svg)


def _bulk_steps():
    return [
        ("Load YAML config (paths/constants/parameters) and dataset params", "#e8f4ff"),
        ("Initialize BulkGEPGenerator (merged scRNA-seq + optional SCT ref)", "#e8f4ff"),
        ("Generate cell-type proportions (sampling_method_prop)", "#fff5e6"),
        ("Sample per-type expression profiles from scRNA-seq", "#fff5e6"),
        ('Mix into bulk GEPs (simu_method = "mul") and write .h5ad', "#e6fff0"),
        ("Optional: TCGA-guided filtering", "#fde8ef"),
        ("Optional: PCA validation plots / artifacts", "#fde8ef"),
    ]


def _sct_steps():
    return [
        ("Load YAML config (paths/constants/parameters) and dataset params", "#e8f4ff"),
        ("Initialize SingleCellTypeGEPGenerator (merged scRNA-seq)", "#e8f4ff"),
        ("For each cell type/subtype: sample n_base cells", "#fff5e6"),
        ('Aggregate into SCT profiles (simu_method = "ave")', "#e6fff0"),
        ("Write SCT .h5ad output", "#e6fff0"),
        ("Optional: downstream use as SCT reference for mixed bulk runs", "#fde8ef"),
    ]


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    bulk_svg = _svg_flow(
        "SimuTME Workflow: Mixed Bulk RNA-seq Simulation",
        _bulk_steps(),
        width=1200,
        height=980,
    )
    (out_dir / "workflow_bulk.svg").write_text(bulk_svg, encoding="utf-8")

    sct_svg = _svg_flow(
        "SimuTME Workflow: sctGEP Simulation",
        _sct_steps(),
        width=1200,
        height=880,
    )
    (out_dir / "workflow_sct.svg").write_text(sct_svg, encoding="utf-8")


if __name__ == "__main__":
    main()
