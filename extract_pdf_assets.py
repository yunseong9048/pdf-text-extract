"""Extract readable PDF text and visual assets for v101.

The workflow is intentionally conservative:
- text is extracted in page order with block coordinates for traceability;
- every page is rendered to PNG so vector figures/tables are never lost;
- embedded images and image-block crops are saved separately when available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import html
import textwrap
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fitz
from equation_text import horizontal_rules, math_candidate, normalize_pdf_symbols, raw_characters, reconstruct_equation
from table import extract_table_rows, process_tables, rows_to_markdown, table_caption_match

# 처음 읽을 때는 맨 아래 main() → extract_pdf() → text_blocks() 순서로 보세요.
# 처리 흐름: PDF 선택 → 본문/수식/표 추출 → 읽기 순서 정렬 → 이미지 저장 → 결과 검증.
# PDF의 좌표는 왼쪽 위가 원점이며, x는 오른쪽으로, y는 아래쪽으로 커집니다.
# bbox = [왼쪽 x, 위쪽 y, 오른쪽 x, 아래쪽 y]로 나타낸 사각형 영역입니다.
# block은 PDF가 묶어 둔 글자 영역으로, 논문의 실제 문단과 항상 같지는 않습니다.
# span은 같은 글꼴·크기 등의 속성을 가진 글자 묶음입니다.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "CatalystHub Zr precursor papers_all" / "Paper"
DEFAULT_OUTPUT = ROOT / "outputs" / "v101_pdf_text_extract"


@dataclass
class ExtractedImage:
    # 이미지 파일 하나의 출처와 위치 정보입니다. 나중에 JSON으로 저장합니다.
    page: int
    kind: str
    path: str
    width: int | None = None
    height: int | None = None
    bbox: list[float] | None = None
    xref: int | None = None
    sha256: str | None = None
    figure: str | None = None
    caption: str | None = None
    note: str | None = None


@dataclass
class PaperSummary:
    # 논문 한 편의 처리 결과입니다. manifest.json과 summary.csv에도 사용합니다.
    paper_id: str
    source_pdf: str
    source_sha256: str
    pages: int
    text_chars: int
    text_blocks: int
    rendered_pages: int
    embedded_images: int
    image_crops: int
    errors: list[str]


def sha256_file(path: Path) -> str:
    # 파일 내용의 지문을 계산해, 검증할 때 원본 PDF가 바뀌었는지 확인합니다.
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", value.strip())
    return value.strip("._") or "paper"


def normalize_heading_spacing(value: str) -> str:
    value = re.sub(r"(?m)^■\s*(?=\w)", "■ ", value)
    return re.sub(r"(?<!\n)\n(?=■ )", "\n\n", value)


def reflow_word_separated_lines(value: str) -> str:
    """Join artificial one-word lines while preserving reference boundaries."""
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) < 6:
        return value
    tokens = [token for line in lines for token in line.split()]
    if not tokens:
        return value
    one_word_lines = sum(len(line.split()) == 1 for line in lines) / len(lines)
    single_char_tokens = sum(len(token) == 1 for token in tokens) / len(tokens)
    # Apply only when the PDF text layer has split most words onto their own
    # lines; labels such as "A R T I C L E" are intentionally left untouched.
    if one_word_lines < .55 or single_char_tokens > .30:
        return value
    joined = re.sub(r"\s+", " ", " ".join(lines)).strip()
    # Keep numbered bibliography entries on separate lines when the same PDF
    # text block contains several references; ordinary inline citations remain.
    entry_pattern = r"\[\d{1,3}\]\s+[A-Z](?:\.|-)"
    if len(re.findall(entry_pattern, joined)) >= 2:
        joined = re.sub(r"\s+(?=\[\d{1,3}\]\s+[A-Z](?:\.|-))", "\n", joined)
    return "\n".join(
        textwrap.fill(line, width=100, break_long_words=False, break_on_hyphens=False)
        for line in joined.splitlines()
    )




def document_word_forms(doc: fitz.Document) -> set[str]:
    """Use only complete, unwrapped tokens actually printed in this PDF."""
    # 외부 사전 대신 이 PDF의 표기를 수집합니다. 줄을 넘어서 단어를 만들지는 않습니다.
    # high-aspect-ratio는 한 덩어리로, 줄 끝 techno-는 하이픈을 포함해 수집합니다.
    forms: set[str] = set()
    for page in doc:
        for block in page.get_text("blocks"):
            if block[6] == 0:
                forms.update(re.findall(r"[A-Za-z]+(?:[-\u00ad][A-Za-z]*)*",
                                        normalize_pdf_symbols(block[4])))
    return forms


def dehyphenate_wrapped_words(value: str, forms: set[str] | None = None,
                             decisions: list[dict[str, str]] | None = None) -> str:
    # forms는 같은 PDF에서 관찰한 표기이고, decisions는 처리 근거를 남길 목록입니다.
    # soft hyphen은 줄바꿈용 표시입니다. 일반 '-'는 붙인 표기만 확인될 때 제거합니다.
    # 근거가 없거나 두 표기가 모두 있으면 '-'를 남겨 합성어를 잘못 바꾸지 않습니다.
    def join_if_supported(match: re.Match[str]) -> str:
        left, hyphen, right = match.group(1), match.group(2), match.group(3)
        joined, retained = left + right, left + "-" + right
        if hyphen == "\u00ad":
            result, reason = joined, "explicit_soft_hyphen"
        elif forms and joined in forms and retained not in forms:
            result, reason = joined, "document_unwrapped_form"
        else:
            result, reason = retained, "hyphen_preserved_uncertain"
            if forms and retained in forms:
                reason = "document_hyphenated_form" if joined not in forms else "conflicting_document_forms"
        if decisions is not None:
            decisions.append(dict(source=match.group(0), result=result, reason=reason))
        return result

    # 정규식의 \n 하나만 연결 대상으로 삼아, 빈 줄로 구분된 문단은 합치지 않습니다.
    return re.sub(r"\b([A-Za-z]+(?:-[A-Za-z]+)*)([-\u00ad])[ \t]*\n[ \t]*([a-z][A-Za-z]*(?:-[A-Za-z]+)*)\b",
                  join_if_supported, value)


def rel(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))


def figure_captions(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 본문 속 'Figure 3' 언급과 구별하기 위해, 블록 시작의 캡션 형식을 찾습니다.
    captions = []
    pattern = re.compile(r"(?i)^\s*(Figure|Fig\.?|Scheme)[ \t]+((?:S\s*)?\d+)\s*[.:]")
    for block in blocks:
        if not block.get("caption_candidate", True):
            continue
        match = pattern.search(block["text"])
        if match:
            number = re.sub(r"\s+", "", match.group(2))
            label = "Scheme" if match.group(1).lower() == "scheme" else "Figure"
            captions.append({"figure": f"{label} {number}", "text": block["text"].strip(), "bbox": block["bbox"]})
    return captions


def _merge_graphic_boxes(boxes: list[fitz.Rect], gap: float = 12) -> list[fitz.Rect]:
    """Cluster nearby vector/image rectangles into layout-level graphic regions."""
    clusters: list[fitz.Rect] = []
    for box in sorted(boxes, key=lambda r: (r.y0, r.x0)):
        merged = box
        keep: list[fitz.Rect] = []
        changed = True
        while changed:
            changed = False
            for current in clusters + keep:
                if (merged + (-gap, -gap, gap, gap)).intersects(current):
                    merged |= current
                    if current in clusters:
                        clusters.remove(current)
                    if current in keep:
                        keep.remove(current)
                    changed = True
                    break
        clusters.append(merged)
    return clusters


def suppress_figure_text(page: fitz.Page, blocks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exclude short text over figure graphics from prose, keeping source blocks in JSON."""
    captions = figure_captions(blocks)
    if not captions:
        return blocks, []
    page_area = page.rect.width * page.rect.height
    graphics: list[fitz.Rect] = []
    for drawing in page.get_drawings():
        box = fitz.Rect(drawing["rect"])
        if box.width >= 2 and box.height >= 2 and box.get_area() < page_area * .55:
            graphics.append(box)
    for image in page.get_image_info():
        box = fitz.Rect(image["bbox"])
        if box.width >= 12 and box.height >= 12:
            graphics.append(box)
    warnings: list[dict[str, Any]] = []
    table_zones: list[fitz.Rect] = []
    for block in blocks:
        if not table_caption_match(block.get("text", "")):
            continue
        cap = fitz.Rect(block["bbox"])
        x1 = page.rect.width * .905 if cap.x0 > page.rect.width / 2 or cap.x0 < page.rect.width * .3 else page.rect.width * .485
        next_body = [item["bbox"][1] for item in blocks
                     if item is not block and item["bbox"][1] > cap.y1 + 4
                     and cap.x0 <= item["bbox"][0] < x1
                     and len(re.findall(r"[A-Za-z]{3,}", item.get("text", "").splitlines()[0])) >= 5]
        bottom = min(next_body + [min(page.rect.height * .94, cap.y1 + 200)])
        table_zones.append(fitz.Rect(cap.x0, cap.y1, x1, bottom))

    for caption in captions:
        cap = fitz.Rect(caption["bbox"])
        local_graphics = [region for region in graphics
                          if region.y0 < cap.y0 and region.y1 > cap.y0 - 420
                          and (min(cap.x1, region.x1) - max(cap.x0, region.x0) >= 12
                               or abs(region.x0 + region.x1 - cap.x0 - cap.x1) <= 120)]
        clusters = _merge_graphic_boxes(local_graphics)
        nearby = []
        for region in clusters:
            gap = cap.y0 - region.y1
            horizontal_overlap = max(0.0, min(cap.x1, region.x1) - max(cap.x0, region.x0))
            horizontally_near = horizontal_overlap >= min(24.0, cap.width * .12) or abs(region.x0 + region.x1 - cap.x0 - cap.x1) <= 100
            if -4 <= gap <= 360 and horizontally_near:
                nearby.append((gap, region))
        if not nearby:
            continue
        # The closest substantial graphic cluster is the figure immediately above this caption.
        nearby.sort(key=lambda pair: pair[0])
        nearest_gap, region = nearby[0]
        region = fitz.Rect(region)
        for gap, candidate in nearby[1:]:
            if gap - nearest_gap <= 70 and candidate.y1 >= region.y0 - 70:
                region |= candidate
        region = (region + (-24, -24, 24, 18)) & page.rect
        if (region.width < 40 or region.height < 25 or region.get_area() < page_area * .002
                or region.get_area() > page_area * .65):
            continue
        for block in blocks:
            block_text = block.get("text", "")
            if (block.get("excluded_from_body")
                    or re.match(r"(?i)^\s*(?:figure|fig\.?|scheme)\s+\d+", block_text)
                    or table_caption_match(block_text)):
                continue
            text = block.get("text", "").strip()
            if not text or len(text) > 180:
                continue
            words = re.findall(r"[A-Za-z]{2,}|[0-9]+", text)
            if len(words) > 15 or (len(words) >= 8 and re.search(r"[.!?]\s*$", text)):
                continue
            box = fitz.Rect(block["bbox"])
            if any(zone.contains(box) for zone in table_zones):
                continue
            overlap = (box & region).get_area() / max(box.get_area(), 1.0)
            if overlap < .45 and not region.contains(box.tl) and not region.contains(box.br):
                continue
            block["excluded_from_body"] = True
            block["exclusion_reason"] = "short text overlaps vector/image region associated with a figure caption"
            warnings.append({"code": "figure_text_suppressed", "block_no": block.get("block_no"),
                             "figure": caption["figure"], "text": text, "bbox": block["bbox"]})
    return blocks, warnings


def match_figure(bbox: fitz.Rect, captions: list[dict[str, Any]]) -> dict[str, Any] | None:
    # 그림과 캡션의 가로 겹침·세로 거리를 비교합니다. 번호 연결은 위치 기반 추정입니다.
    best: tuple[float, dict[str, Any]] | None = None
    for caption in captions:
        cap = fitz.Rect(caption["bbox"])
        overlap = max(0.0, min(bbox.x1, cap.x1) - max(bbox.x0, cap.x0))
        if overlap == 0 and abs(bbox.x0 + bbox.x1 - cap.x0 - cap.x1) / 2 > 48:
            continue
        vertical_gap = max(0.0, max(bbox.y0, cap.y0) - min(bbox.y1, cap.y1))
        if vertical_gap > 120:
            continue
        # Journal figure captions are below the graphic. A nearby caption above
        # it belongs to the previous figure, even when its gap is smaller.
        score = vertical_gap + (0 if overlap else 48) + (1000 if cap.y0 < bbox.y0 else 0)
        if best is None or score < best[0]:
            best = (score, caption)
    return best[1] if best else None


def _dense_table_block(block: dict[str, Any]) -> bool:
    """Recognize multi-row table text blocks that extend slightly past the crop box."""
    text = str(block.get("text", ""))
    lines = [line for line in text.splitlines() if line.strip()]
    tokens = re.findall(r"\S+", text)
    numeric = re.findall(r"(?<![A-Za-z])(?:\d+(?:[.,]\d+)?|[<>≤≥])", text)
    return len(lines) >= 4 and len(numeric) >= 6 and len(numeric) / max(1, len(tokens)) >= .2


def _overlaps_table_box(block: dict[str, Any], box: fitz.Rect) -> bool:
    """Include slight PDF bbox overhangs for compact headings and dense table rows."""
    rect = fitz.Rect(block["bbox"])
    overlap = (rect & box).get_area() / max(rect.get_area(), 1.0)
    if overlap < .75:
        return False
    text = str(block.get("text", ""))
    height = rect.height
    compact_header = height <= 16 and len(text) <= 400 and len(text.split()) >= 4
    return compact_header or _dense_table_block(block)


def image_stem(page_no: int, figure: str | None, kind: str, index: int) -> str:
    # 그림 번호와 PDF 페이지 번호로 파일명을 만듭니다. p004는 네 번째 PDF 페이지입니다.
    if figure:
        label = "scheme" if figure.startswith("Scheme") else "figure"
        number = re.sub(r"[^0-9A-Za-z]", "", figure.removeprefix("Figure").removeprefix("Scheme"))
        prefix = f"{label}{number}_p{page_no:03d}"
    else:
        prefix = f"image_p{page_no:03d}"
    return f"{prefix}_{kind}{index:02d}"


def archive_obsolete_images(image_dir: Path, images: list[ExtractedImage], base_dir: Path) -> None:
    # 재추출 후 사용하지 않는 기존 자동 생성 이미지는 삭제하지 않고 별도로 옮깁니다.
    referenced = {str((base_dir / item.path).resolve()) for item in images if item.path}
    for path in image_dir.iterdir():
        generated = re.match(r"^(?:p\d{3}_|(?:figure\d+|scheme\d+|image)_p\d{3}_)", path.name)
        if path.is_file() and generated and str(path.resolve()) not in referenced:
            archive = base_dir / "superseded_images" / image_dir.name
            archive.mkdir(parents=True, exist_ok=True)
            path.replace(archive / path.name)


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "texts": out_dir / "texts",
        "json": out_dir / "json",
        "pages": out_dir / "pages",
        "images": out_dir / "images",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs








def save_region(page: fitz.Page, box: fitz.Rect, dest: Path, output: Path, zoom: float,
                kind: str, **metadata: Any) -> ExtractedImage:
    # 글자 가장자리가 잘리지 않도록 여백을 조금 더하되, 페이지 밖으로 나가지는 않습니다.
    box = (box + (-2, -2, 2, 2)) & page.rect
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=box, alpha=False)
    pix.save(dest)
    return ExtractedImage(page=page.number + 1, kind=kind, path=rel(dest, output),
                          width=pix.width, height=pix.height, bbox=list(box),
                          sha256=sha256_file(dest), **metadata)




def extract_structures(page: fitz.Page, blocks: list[dict[str, Any]], image_dir: Path,
                       output: Path, zoom: float) -> tuple[list[dict[str, Any]], list[ExtractedImage], list[dict[str, Any]]]:
    # 반환값: 처리된 글자 블록, 저장한 이미지 목록, 사람이 확인해야 할 경고 목록.
    # 수식/표를 묶어도 source_block_nos와 native_text에 원본 출처를 보존합니다.
    images: list[ExtractedImage] = []
    warnings: list[dict[str, Any]] = []
    blocks, figure_warnings = suppress_figure_text(page, blocks)
    warnings.extend(figure_warnings)
    blocks, table_warnings = process_tables(blocks)
    warnings.extend(table_warnings)
    native = {b[5]: b[4] for b in page.get_text("blocks") if b[6] == 0}
    chars = raw_characters(page)
    rules = horizontal_rules(page)

    # 표 캡션 아래의 같은 단에서 탐색하고, 다음 본문이 시작되면 표 영역을 끝냅니다.
    for caption in list(blocks):
        match = table_caption_match(caption["text"])
        if not match or not caption.get("caption_candidate", True):
            continue
        cap = fitz.Rect(caption["bbox"])
        wide = cap.x0 < page.rect.width * .3 and cap.x1 > page.rect.width * .6
        x0 = cap.x0
        base_x1 = page.rect.width * .905 if wide or cap.x0 > page.rect.width / 2 else page.rect.width * .485
        # A left-column caption can head a table that extends beyond the column.
        # Infer its right edge from compact text rows directly beneath the caption.
        compact_rows = [b for b in blocks if cap.y1 - 1 <= b["bbox"][1] <= cap.y1 + 180
                        and b["bbox"][3] - b["bbox"][1] <= 30
                        and b["bbox"][0] >= x0 - 8
                        and b["bbox"][0] < page.rect.width * .72]
        dense_table_rows = [b for b in blocks if cap.y1 - 1 <= b["bbox"][1] <= cap.y1 + 180
                            and b["bbox"][0] >= x0 - 8
                            and b["bbox"][0] < page.rect.width * .72
                            and _dense_table_block(b)]
        extent_blocks = compact_rows + dense_table_rows
        if extent_blocks and not wide and cap.x0 < page.rect.width / 2:
            base_x1 = max(base_x1, max(b["bbox"][2] for b in extent_blocks) + 3)
        x1 = min(page.rect.width * .94, base_x1)
        following = [b["bbox"][1] for b in blocks if b["bbox"][1] > cap.y1 + 4
                     and x0 <= b["bbox"][0] < x1
                     and b["bbox"][3] - b["bbox"][1] >= 16
                     and len(re.findall(r"[A-Za-z]{3,}", b["text"].splitlines()[0])) >= 5]
        following_tables = [b["bbox"][1] for b in blocks if b is not caption
                            and table_caption_match(b["text"])
                            and b["bbox"][1] > cap.y1 + 3
                            and x0 <= b["bbox"][0] < x1]
        following_tables = [y - 4 for y in following_tables]
        default_bottom = min(page.rect.height * .94, cap.y1 + 200)
        bottom = min(following + following_tables + [default_bottom])
        if dense_table_rows:
            dense_bottom = max(b["bbox"][3] for b in dense_table_rows)
            table_limit = min(following_tables + [page.rect.height * .94])
            bottom = min(max(bottom, dense_bottom), table_limit)
        box = fitz.Rect(x0, cap.y1 + 1, x1, bottom - 1)
        if box.height < 12 or box.width < 12:
            continue
        selected = [b for b in blocks if b is not caption and not b.get("excluded_from_body")
                    and (box.contains(fitz.Rect(b["bbox"])) or _overlaps_table_box(b, box))]
        asset = save_region(page, box, image_dir / f"table{match[1]}_p{page.number+1:03d}.png",
                            output, zoom, "table_region")
        images.append(asset)
        rows, table_confidence, table_error, table_footnotes = extract_table_rows(page, box)
        if table_error:
            warnings.append(dict(code="table_detection_failed", detail=table_error, bbox=list(box)))
        if selected:
            ids = [b["block_no"] for b in selected]
            native_text = "".join(native.get(i, "") for i in ids)
            # Some PDF fonts omit the micro sign from word-level geometry, even
            # though the raw source block retains its unmapped glyph. Restore it
            # only when both the header and raw source provide that evidence.
            if rows and "\x02M" in native_text:
                header = [cell.lower() for cell in rows[0]]
                lod_columns = [i for i, cell in enumerate(header)
                               if "limit of detection" in cell]
                for column in lod_columns:
                    for row in rows[1:]:
                        if column < len(row) and re.fullmatch(r"\s*\d+(?:\.\d+)?\s*M\s*", row[column]):
                            row[column] = re.sub(r"\bM\b", "µM", row[column])
            table_markdown = rows_to_markdown(rows) if rows else None
            rendered_table = table_markdown if table_markdown else f"[TABLE_CHECK: {asset.path}]\n{native_text}"
            if table_markdown and table_footnotes:
                rendered_table += "\n\n" + "\n".join(table_footnotes)
            merged = dict(selected[0], bbox=list(box), source_block_nos=ids,
                          native_text=native_text, kind="table",
                          status="auto_formatted_needs_review" if rows else "needs_review",
                          region_image=asset.path, candidate_rows=rows,
                          table_confidence=table_confidence, table_markdown=table_markdown,
                          table_footnotes=table_footnotes,
                          text=rendered_table)
            blocks = [b for b in blocks if b not in selected] + [merged]
        warnings.append(dict(code="table_auto_formatted_needs_review" if rows else "table_needs_review",
                             image=asset.path, bbox=list(box), confidence=table_confidence))

    # 같은 단에서 세로로 가까운 짧은 블록들을 하나의 수식 후보로 묶습니다.
    candidates = [b for b in blocks if "kind" not in b and not b.get("excluded_from_body") and math_candidate(b)]
    groups: list[list[dict[str, Any]]] = []
    for block in sorted(candidates, key=lambda b: b["bbox"][1]):
        box = fitz.Rect(block["bbox"])
        for group in reversed(groups):
            other = fitz.Rect(group[0]["bbox"])
            for member in group[1:]:
                other |= fitz.Rect(member["bbox"])
            same_column = (box.x0 + box.x1 < page.rect.width) == (other.x0 + other.x1 < page.rect.width)
            union = box | other
            if same_column and box.y0 <= other.y1 + 7 and union.height < 58:
                group.append(block)
                break
        else:
            groups.append([block])
    for group in groups:
        raw = "\n".join(b["text"] for b in group)
        if not re.search(r"[=→↔⇌]", raw):
            continue
        # A sequence of individually short caption/table fragments can look
        # mathematical after grouping (for example, statistical annotations
        # or reaction products inside a table). Keep long prose and labeled
        # layout regions in their original reading order instead.
        word_count = len(re.findall(r"[A-Za-z]{3,}", raw))
        prose_word_count = len(re.findall(
            r"(?i)\b(?:and|after|before|during|into|from|with|for|the|of|in|by|is|was|were|indicates|respectively)\b",
            raw,
        ))
        if (word_count > 12 or len(raw) > 260
                or (word_count >= 6 and prose_word_count >= 3)
                or re.search(r"(?i)\b(?:fig(?:ure)?\.?|table|scheme|gpc)\b", raw)):
            continue
        # Prose containing an inline operator must not become an equation.
        if any(not math_candidate(b) for b in group):
            continue
        ids = [b["block_no"] for b in group]
        box = fitz.Rect(group[0]["bbox"])
        for b in group[1:]:
            box |= fitz.Rect(b["bbox"])
        content = [c for n in ids for c in chars.get(n, [])]
        reconstructed = reconstruct_equation(content, rules)
        # 복원 성공 여부와 관계없이 원본 이미지를 남기고 검토 대상으로 표시합니다.
        asset = save_region(page, box, image_dir / f"equation_p{page.number+1:03d}_b{min(ids):03d}.png",
                            output, zoom, "equation_region")
        images.append(asset)
        native_text = "".join(native.get(n, "") for n in ids)
        if reconstructed:
            equation_text = reconstructed
            status = "reconstructed_needs_review"
            warning_code = "equation_reconstruction_needs_review"
        else:
            equation_text = f"[EQUATION_REVIEW: {asset.path}]"
            status = "needs_review"
            warning_code = "equation_needs_review"
        merged = dict(group[0], bbox=list(box), kind="equation", source_block_nos=ids,
                      native_text=native_text, status=status, region_image=asset.path,
                      reconstruction=reconstructed, text=equation_text)
        blocks = [b for b in blocks if b not in group] + [merged]
        warnings.append(dict(code=warning_code, image=asset.path, bbox=list(box)))

    for block in blocks:
        if "kind" in block:
            continue
        prefix = re.match(r"(?s)^(.{1,160}?)(,\s*where\b)", block["text"])
        if prefix and prefix[1].count("\n") >= 3 and "=" in prefix[1]:
            box = fitz.Rect(block["bbox"])
            box.y1 = min(box.y1, box.y0 + 25)
            asset = save_region(page, box, image_dir / f"equation_p{page.number+1:03d}_inline{block['block_no']:03d}.png",
                                output, zoom, "equation_region")
            images.append(asset)
            block.update(native_text=native[block["block_no"]], kind="inline_equation",
                         status="needs_review", region_image=asset.path,
                         source_block_nos=[block["block_no"]],
                         text=f"[EQUATION_CHECK: {asset.path}]" + block["text"][len(prefix[1]):])
            warnings.append(dict(code="equation_needs_review", image=asset.path, bbox=list(box)))
        if any(0xe000 <= ord(c) <= 0xf8ff or c == "\ufffd" for c in block["text"]):
            warnings.append(dict(code="unmapped_glyph", block=block["block_no"], bbox=block["bbox"]))
    if not blocks:
        warnings.append(dict(code="no_text_layer", detail="OCR is required; full-page image is preserved."))
    return blocks, images, warnings


def save_figure_regions(page: fitz.Page, captions: list[dict[str, Any]], blocks: list[dict[str, Any]],
                        image_dir: Path, output: Path, zoom: float, warnings: list[dict[str, Any]]) -> list[ExtractedImage]:
    # 벡터 그래프는 내장 이미지 추출로 얻지 못하므로, 캡션 주변을 페이지에서 렌더링합니다.
    # 경계를 추정하지 못하면 그림 일부를 잃지 않도록 전체 페이지를 저장하고 경고합니다.
    results = []
    rules = horizontal_rules(page)
    # 수평선이 그래프의 아래쪽 축일 수 있으므로 실제 그래픽 범위도 함께 검사합니다.
    graphics = [fitz.Rect(d["rect"]) for d in page.get_drawings() if d["rect"].height > 3]
    graphics.extend(fitz.Rect(i["bbox"]) for i in page.get_image_info())
    for index, caption in enumerate(captions, 1):
        cap = fitz.Rect(caption["bbox"])
        possible = [(a, b, y) for a, b, y in rules if y < cap.y0 - 3
                    and min(b, cap.x1) - max(a, cap.x0) >= cap.width * .8]
        box = None
        if possible:
            a, b, y = max(possible, key=lambda r: r[2])
            wide = cap.x0 < page.rect.width * .3 and cap.x1 > page.rect.width * .7
            box = fitz.Rect(a if wide else cap.x0, y + 1, b if wide else cap.x1, cap.y1)
            prose = [t for t in blocks if len(t["text"]) > 100 and t["bbox"][3] < cap.y0
                     and box.contains(fitz.Rect(t["bbox"]))]
            if prose or cap.y0 - y < 15:
                box = None
            if box is not None:
                preceding = [b["bbox"][3] for b in blocks
                             if b["bbox"][3] < cap.y0 and len(b["text"]) > 100
                             and b["bbox"][0] < cap.x1 and b["bbox"][2] > cap.x0]
                preceding.extend(c["bbox"][3] for c in captions if c["bbox"][3] < cap.y0
                                 and c["bbox"][0] < cap.x1 and c["bbox"][2] > cap.x0)
                top = max(preceding, default=0)
                nearby = [g for g in graphics if g.y1 > top and g.y0 < cap.y0
                          and g.x0 < cap.x1 and g.x1 > cap.x0]
                # 그래픽이 후보 영역 밖에 남거나 근거가 없으면 crop을 확정하지 않습니다.
                if not nearby or any(not (box + (-2, -2, 2, 2)).contains(g) for g in nearby):
                    box = None
        if box is None:
            box = page.rect
            kind = "figure_page_fallback"
            warnings.append(dict(code="figure_region_uncertain", figure=caption["figure"],
                                 detail="Full page saved under the figure name."))
        else:
            kind = "figure_region"
        stem = image_stem(page.number + 1, caption["figure"], "full" if kind == "figure_region" else "page", index)
        results.append(save_region(page, box, image_dir / f"{stem}.png", output, zoom, kind,
                                   figure=caption["figure"], caption=caption["text"],
                                   note="Automatically located; consult the page for boundary verification."))
    return results


def citation_spans(line: dict[str, Any]) -> list[dict[str, Any]]:
    # 파란색·작은 글씨·위치·숫자 형식을 함께 보고 참고문헌 번호 후보를 찾습니다.
    # 해당 부분을 [CITATION:번호]로 바꾸기 위한 문자열 위치와 표시값을 반환합니다.
    spans = line.get("spans", [])
    if not spans:
        return []
    largest_size = max(float(span.get("size", 0)) for span in spans)
    base_spans = [
        span for span in spans
        if int(span.get("color", 0)) == 0 and float(span.get("size", 0)) >= largest_size * 0.85
    ]
    detached = not base_spans and all(
        re.fullmatch(r"[\d\s,;–−-]+", s.get("text", "")) and s.get("size", 100) < 9
        for s in spans
    )
    if not base_spans and not detached:
        return []
    baseline = sorted(float(span.get("origin", [0, span["bbox"][3]])[1]) for span in base_spans)
    baseline_y = baseline[len(baseline) // 2] if baseline else 0

    candidates: list[int] = []
    for index, span in enumerate(spans):
        color = int(span.get("color", 0))
        red, green, blue = (color >> 16) & 255, (color >> 8) & 255, color & 255
        value = span.get("text", "").strip()
        is_blue = blue > red * 1.35 and blue > green * 1.15
        is_small = float(span.get("size", 0)) < largest_size * 0.85
        is_raised = float(span.get("origin", [0, span["bbox"][3]])[1]) < baseline_y - 1
        leading_reference = all(re.fullmatch(r"[\d\s,;–−-]*", s.get("text", ""))
                                for s in spans[:index])
        if is_blue and ((is_small and (is_raised or leading_reference)) or detached) and re.fullmatch(r"\d+(?:[,;–−-]\d+)*", value):
            candidates.append(index)

    groups: list[list[int]] = []
    for index in candidates:
        if groups:
            between = spans[groups[-1][-1] + 1:index]
            if between and all(re.fullmatch(r"[\s,;:–−-]*", item.get("text", "")) for item in between):
                groups[-1].append(index)
                continue
        groups.append([index])

    results = []
    for group in groups:
        first, last = group[0], group[-1]
        value = "".join(span.get("text", "") for span in spans[first:last + 1]).strip()
        value = re.sub(r"\s+", "", value)
        boxes = [fitz.Rect(span["bbox"]) for span in spans[first:last + 1]]
        box = fitz.Rect(min(b.x0 for b in boxes), min(b.y0 for b in boxes), max(b.x1 for b in boxes), max(b.y1 for b in boxes))
        results.append(
            {
                "text": value,
                "marker": f"[CITATION:{value}]",
                "bbox": [round(float(v), 2) for v in box],
                "color": f"#{int(spans[first]['color']):06X}",
                "kind": "superscript_reference",
                "_start": sum(len(span.get("text", "")) for span in spans[:first]),
                "_end": sum(len(span.get("text", "")) for span in spans[:last + 1]),
            }
        )
    return results


def script_spans(line: dict[str, Any], citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 인용 번호와 겹치지 않는 작은 숫자를 위치에 따라 위첨자/아래첨자로 바꿉니다.
    spans = line.get("spans", [])
    if not spans:
        return []
    largest = max(s["size"] for s in spans)
    base = [s["origin"][1] for s in spans if s["size"] >= largest * .85]
    baseline = sorted(base)[len(base) // 2]
    sub = str.maketrans("0123456789+-−", "₀₁₂₃₄₅₆₇₈₉₊₋₋")
    sup = str.maketrans("0123456789+-−", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻")
    offset = 0
    results = []
    for span in spans:
        value = span["text"]
        end = offset + len(value)
        covered = any(c["_start"] < end and c["_end"] > offset for c in citations)
        # Blue numbers are references, never chemical or mathematical indices.
        color = span.get("color", 0)
        black = ((color >> 16) & 255) == (color & 255)
        dy = span["origin"][1] - baseline
        if (not covered and black and span["size"] < largest * .85
                and abs(dy) > 1 and re.fullmatch(r"\s*[+−-]?\d+[+−-]?\s*", value)):
            results.append({"_start": offset, "_end": end,
                            "marker": value.translate(sub if dy > 0 else sup)})
        offset = end
    return results


def text_blocks(page: fitz.Page, forms: set[str] | None = None) -> list[dict[str, Any]]:
    # 일반 본문 처리: 원래 블록 내부 순서를 유지하면서 인용·첨자·합자·하이픈을 정리합니다.
    # blocks는 원문 문자열, dict는 글꼴/색상/위치 정보이며 bbox로 서로 대응시킵니다.
    blocks: list[dict[str, Any]] = []
    dict_blocks = page.get_text("dict").get("blocks", [])

    for block in page.get_text("blocks", sort=True):
        x0, y0, x1, y1, text, block_no, block_type = block[:7]
        if block_type != 0:
            continue
        clean = normalize_heading_spacing(text.replace("\x00", "").strip())
        if not clean or (text.startswith("Downloaded from ") and y1 - y0 > (x1 - x0) * 3):
            continue
        bbox_key = tuple(round(float(v), 2) for v in (x0, y0, x1, y1))
        raw_block = next(
            (item for item in dict_blocks if item.get("type") == 0 and tuple(round(float(v), 2) for v in item["bbox"]) == bbox_key),
            None,
        )
        first_span = next((s for l in (raw_block or {}).get("lines", [])
                           for s in l.get("spans", []) if s["text"].strip()), {})
        citations: list[dict[str, Any]] = []
        if raw_block:
            reference_entries = re.findall(r"(?m)^\s*\[\d{1,3}\]\s*$", clean)
            in_reference_list = len(reference_entries) >= 2
            search_from = 0
            for line in raw_block.get("lines", []):
                line_citations = [] if in_reference_list else citation_spans(line)
                annotations = [] if in_reference_list else line_citations + script_spans(line, line_citations)
                if not annotations:
                    continue
                line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                line_parts = []
                cursor = 0
                for annotation in sorted(annotations, key=lambda a: a["_start"]):
                    start, end = annotation["_start"], annotation["_end"]
                    line_parts.extend((line_text[cursor:start], annotation["marker"]))
                    cursor = end
                citations.extend({k: v for k, v in c.items() if not k.startswith("_")}
                                 for c in line_citations)
                line_parts.append(line_text[cursor:])
                annotated_line = "".join(line_parts)
                # Search only after the previous line. A short line such as
                # "8" can otherwise match a digit inside a citation marker
                # inserted for an earlier line in the same text block.
                line_start = clean.find(line_text, search_from)
                if line_start < 0:
                    line_start = clean.find(line_text)
                if line_start >= 0:
                    clean = (clean[:line_start] + annotated_line
                             + clean[line_start + len(line_text):])
                    search_from = line_start + len(annotated_line)
        hyphenation: list[dict[str, str]] = []
        clean = dehyphenate_wrapped_words(normalize_pdf_symbols(clean), forms, hyphenation)
        clean = reflow_word_separated_lines(clean)
        blocks.append(
            {
                "block_no": int(block_no),
                "bbox": [round(float(v), 2) for v in (x0, y0, x1, y1)],
                "text": clean,
                "hyphenation": hyphenation,
                "citations": citations,
                "caption_candidate": first_span.get("color", 0) == 0,
            }
        )
    return blocks


def reading_order(blocks: list[dict[str, Any]], page: fitz.Page) -> list[dict[str, Any]]:
    """Read columns within bands separated by full-width text/captions."""
    # 페이지 전체를 y좌표로만 정렬하면 좌우 단의 문장이 섞일 수 있습니다.
    # 전체 너비 블록으로 구간을 나눈 뒤, 각 구간에서 왼쪽 단 → 오른쪽 단 순으로 읽습니다.
    # 머리말/꼬리말과 단 구분은 좌표 비율 기반 추정이라 모든 레이아웃을 보장하지 않습니다.
    width, height = page.rect.width, page.rect.height
    by_position = lambda b: (b["bbox"][1], b["bbox"][0])
    headers = [b for b in blocks if b["bbox"][3] < height * .075]
    footers = [b for b in blocks if b["bbox"][1] > height * .94]
    body = [b for b in blocks if b not in headers and b not in footers]
    wide = sorted([b for b in body if b["bbox"][0] < width * .3
                   and b["bbox"][2] > width * .7], key=by_position)
    remaining = [b for b in body if b not in wide]
    result = sorted(headers, key=by_position)

    def columns(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # 좌우에 실제로 떨어진 글자 영역이 세로로 겹쳐야 두 단으로 판단합니다.
        # 중앙에 놓인 수식 하나만으로 한 단 문서를 두 단으로 나누지 않습니다.
        left = [b for b in items if b["bbox"][2] <= width / 2]
        right = [b for b in items if b["bbox"][0] >= width / 2]
        two_columns = any(min(a["bbox"][3], b["bbox"][3])
                          - max(a["bbox"][1], b["bbox"][1]) > 12
                          for a in left for b in right)
        if not two_columns:
            return sorted(items, key=by_position)
        return sorted(items, key=lambda b: (
            (b["bbox"][0] + b["bbox"][2]) / 2 >= width / 2,
            *by_position(b),
        ))

    for band in wide:
        above = [b for b in remaining if b["bbox"][1] < band["bbox"][1]]
        result.extend(columns(above))
        result.append(band)
        remaining = [b for b in remaining if b not in above]
    return result + columns(remaining) + sorted(footers, key=by_position)


def render_page(page: fitz.Page, dest: Path, zoom: float) -> tuple[int, int]:
    # 텍스트 추출 결과와 직접 비교할 수 있도록 원본 페이지 전체를 PNG로 남깁니다.
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    pix.save(dest)
    return pix.width, pix.height


def save_embedded_images(
    doc: fitz.Document,
    page: fitz.Page,
    page_no: int,
    image_dir: Path,
    base_dir: Path,
    seen_xrefs: dict[int, ExtractedImage],
    captions: list[dict[str, Any]],
) -> list[ExtractedImage]:
    # xref는 PDF 내부 이미지 객체 번호입니다. 같은 객체가 반복되면 기존 파일을 재사용합니다.
    # 여기서는 내장 이미지 데이터를 꺼내며, 페이지의 벡터 선/글자는 함께 추출되지 않습니다.
    results: list[ExtractedImage] = []
    for img_index, item in enumerate(page.get_images(full=True), start=1):
        xref = int(item[0])
        rects = page.get_image_rects(xref)
        bbox = rects[0] if rects else None
        caption = match_figure(bbox, captions) if bbox else None
        if xref in seen_xrefs:
            prior = seen_xrefs[xref]
            results.append(
                ExtractedImage(
                    page=page_no,
                    kind="embedded_duplicate",
                    path=prior.path,
                    width=prior.width,
                    height=prior.height,
                    xref=xref,
                    sha256=prior.sha256,
                    figure=prior.figure,
                    caption=prior.caption,
                    note=f"same xref as earlier image {xref}",
                )
            )
            continue
        try:
            extracted = doc.extract_image(xref)
            data = extracted["image"]
            ext = extracted.get("ext") or "bin"
            dest = image_dir / f"{image_stem(page_no, caption['figure'] if caption else None, f'xref{xref}_', img_index)}.{ext}"
            dest.write_bytes(data)
            record = ExtractedImage(
                page=page_no,
                kind="embedded",
                path=rel(dest, base_dir),
                width=int(extracted.get("width") or 0),
                height=int(extracted.get("height") or 0),
                bbox=[round(float(v), 2) for v in bbox] if bbox else None,
                xref=xref,
                sha256=sha256_bytes(data),
                figure=caption["figure"] if caption else None,
                caption=caption["text"] if caption else None,
            )
            seen_xrefs[xref] = record
            results.append(record)
        except Exception as exc:  # noqa: BLE001 - continue extracting the paper.
            results.append(
                ExtractedImage(
                    page=page_no,
                    kind="embedded_error",
                    path="",
                    xref=xref,
                    note=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def save_image_block_crops(
    page: fitz.Page,
    page_no: int,
    image_dir: Path,
    base_dir: Path,
    zoom: float,
    captions: list[dict[str, Any]],
) -> list[ExtractedImage]:
    results: list[ExtractedImage] = []
    page_dict = page.get_text("dict")
    crop_index = 0
    for block in page_dict.get("blocks", []):
        if block.get("type") != 1 or "bbox" not in block:
            continue
        bbox = fitz.Rect(block["bbox"]) & page.rect
        if bbox.width < 20 or bbox.height < 20:
            continue
        crop_index += 1
        caption = match_figure(bbox, captions)
        dest = image_dir / f"{image_stem(page_no, caption['figure'] if caption else None, 'crop', crop_index)}.png"
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=bbox, alpha=False)
        pix.save(dest)
        results.append(
            ExtractedImage(
                page=page_no,
                kind="image_block_crop",
                path=rel(dest, base_dir),
                width=pix.width,
                height=pix.height,
                bbox=[round(float(v), 2) for v in bbox],
                sha256=sha256_file(dest),
                figure=caption["figure"] if caption else None,
                caption=caption["text"] if caption else None,
            )
        )
    return results


def validate_output_names(pdfs: list[Path], out_dir: Path) -> None:
    # 추출 전에 전체 파일명을 검사해, 이름 충돌로 다른 논문을 덮어쓰지 않도록 합니다.
    seen: dict[str, Path] = {}
    for pdf in pdfs:
        paper_id = safe_name(pdf.stem)
        key = paper_id.casefold()
        if key in seen:
            raise ValueError(f"Output name collision: {seen[key].name!r}, {pdf.name!r} -> {paper_id!r}")
        seen[key] = pdf
        existing = out_dir / "json" / f"{paper_id}.json"
        if existing.exists():
            previous = json.loads(existing.read_text(encoding="utf-8"))["summary"]["source_pdf"]
            if Path(previous).resolve() != pdf.resolve():
                raise ValueError(f"Output {paper_id!r} already belongs to {previous}; use another output directory")


def extract_pdf(pdf_path: Path, out_dir: Path, page_zoom: float, crop_zoom: float) -> PaperSummary:
    # 실패는 논문 단위로 기록하고 다음 논문을 처리합니다. 예외가 나도 PDF는 닫힙니다.
    try:
        with fitz.open(pdf_path) as doc:
            if doc.needs_pass:
                raise ValueError("Password-protected PDF requires authentication")
            return _extract_pdf(pdf_path, out_dir, page_zoom, crop_zoom, doc)
    except Exception as exc:
        dirs = ensure_dirs(out_dir)
        try:
            source_sha = sha256_file(pdf_path)
        except OSError:
            source_sha = ""
        summary = PaperSummary(safe_name(pdf_path.stem), str(pdf_path), source_sha,
                               0, 0, 0, 0, 0, 0, [f"extraction failed: {type(exc).__name__}: {exc}"])
        (dirs["json"] / f"{summary.paper_id}.json").write_text(
            json.dumps(dict(summary=asdict(summary), pages=[], images=[]), ensure_ascii=False, indent=2),
            encoding="utf-8")
        # 이전 실행의 TXT를 이번 실행의 성공 결과로 오해하지 않도록 비웁니다.
        (dirs["texts"] / f"{summary.paper_id}.txt").write_text("", encoding="utf-8")
        return summary


def _extract_pdf(pdf_path: Path, out_dir: Path, page_zoom: float, crop_zoom: float,
                 doc: fitz.Document) -> PaperSummary:
    # 논문 한 편의 전체 처리 흐름입니다. TXT는 읽기용, JSON은 좌표/출처/경고 확인용입니다.
    dirs = ensure_dirs(out_dir)
    paper_id = safe_name(pdf_path.stem)
    paper_page_dir = dirs["pages"] / paper_id
    paper_image_dir = dirs["images"] / paper_id
    paper_page_dir.mkdir(parents=True, exist_ok=True)
    paper_image_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    pages_json: list[dict[str, Any]] = []
    text_parts: list[str] = []
    images: list[ExtractedImage] = []
    seen_xrefs: dict[int, ExtractedImage] = {}
    source_sha = sha256_file(pdf_path)

    forms = document_word_forms(doc)
    # 문서 전체에서 표기를 먼저 수집해야 뒤쪽 페이지의 표기도 하이픈 판단에 쓸 수 있습니다.
    for page_index, page in enumerate(doc, start=1):
        page_errors: list[str] = []
        page_text_blocks, region_images, warnings = extract_structures(
            page, text_blocks(page, forms), paper_image_dir, out_dir, crop_zoom)
        images.extend(region_images)
        page_text_blocks = reading_order(page_text_blocks, page)
        captions = figure_captions(page_text_blocks)
        images.extend(save_figure_regions(page, captions, page_text_blocks, paper_image_dir, out_dir, crop_zoom, warnings))
        page_text = "\n\n".join(block["text"] for block in page_text_blocks if not block.get("excluded_from_body"))
        text_parts.append(f"\n\n===== PAGE {page_index} =====\n\n{page_text}")

        page_png = paper_page_dir / f"page-{page_index:03d}.png"
        rendered_size: tuple[int, int] | None = None
        try:
            rendered_size = render_page(page, page_png, page_zoom)
        except Exception as exc:  # noqa: BLE001
            msg = f"page {page_index} render failed: {type(exc).__name__}: {exc}"
            errors.append(msg)
            page_errors.append(msg)

        try:
            images.extend(save_embedded_images(doc, page, page_index, paper_image_dir, out_dir, seen_xrefs, captions))
        except Exception as exc:  # noqa: BLE001
            msg = f"page {page_index} embedded image extraction failed: {type(exc).__name__}: {exc}"
            errors.append(msg)
            page_errors.append(msg)

        try:
            images.extend(save_image_block_crops(page, page_index, paper_image_dir, out_dir, crop_zoom, captions))
        except Exception as exc:  # noqa: BLE001
            msg = f"page {page_index} image crop extraction failed: {type(exc).__name__}: {exc}"
            errors.append(msg)
            page_errors.append(msg)

        pages_json.append(
            {
                "page": page_index,
                "width": round(float(page.rect.width), 2),
                "height": round(float(page.rect.height), 2),
                "rendered_png": rel(page_png, out_dir) if page_png.exists() else None,
                "rendered_size": rendered_size,
                "text": page_text,
                "blocks": page_text_blocks,
                "errors": page_errors,
                "warnings": warnings,
            }
        )

    full_text = "".join(text_parts).strip() + "\n"
    (dirs["texts"] / f"{paper_id}.txt").write_text(full_text, encoding="utf-8")

    embedded_count = sum(1 for item in images if item.kind == "embedded")
    crop_count = sum(1 for item in images if item.kind == "image_block_crop")
    summary = PaperSummary(
        paper_id=paper_id,
        source_pdf=str(pdf_path),
        source_sha256=source_sha,
        pages=len(doc),
        text_chars=len(full_text),
        text_blocks=sum(len(page["blocks"]) for page in pages_json),
        rendered_pages=sum(1 for page in pages_json if page["rendered_png"]),
        embedded_images=embedded_count,
        image_crops=crop_count,
        errors=errors,
    )

    payload = {
        "summary": asdict(summary),
        "pages": pages_json,
        "images": [asdict(item) for item in images],
    }
    (dirs["json"] / f"{paper_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not errors:
        archive_obsolete_images(paper_image_dir, images, out_dir)
    return summary


def find_pdfs(input_dir: Path, limit: int | None) -> list[Path]:
    # 입력 폴더 바로 아래의 PDF를 파일명 순으로 선택합니다. 하위 폴더는 탐색하지 않습니다.
    pdfs = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
    if limit is not None:
        pdfs = pdfs[:limit]
    return pdfs


def write_manifest(out_dir: Path, input_dir: Path, summaries: list[PaperSummary]) -> None:
    # 전체 실행의 목차(manifest), 논문별 통계(CSV), 결과 폴더 안내문을 저장합니다.
    manifest = {
        "workflow": "v101_generic_pdf_extraction",
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "papers": [asdict(item) for item in summaries],
        "totals": {
            "papers": len(summaries),
            "pages": sum(item.pages for item in summaries),
            "text_chars": sum(item.text_chars for item in summaries),
            "rendered_pages": sum(item.rendered_pages for item in summaries),
            "embedded_images": sum(item.embedded_images for item in summaries),
            "image_crops": sum(item.image_crops for item in summaries),
            "errors": sum(len(item.errors) for item in summaries),
        },
        "notes": [
            "pages/<paper_id>/page-###.png preserves full pages, including vector graphs and tables.",
            "images/<paper_id>/ contains embedded PDF images and image-block crops when PyMuPDF can identify them.",
            "json/<paper_id>.json stores page text, text block coordinates, image metadata, and extraction errors.",
            "Verified private-use chemistry glyphs: U+E0D5 -> -, U+E0C8 -> =, U+E0C9 -> ≡.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()) if summaries else ["paper_id"])
        writer.writeheader()
        for item in summaries:
            row = asdict(item)
            row["errors"] = " | ".join(item.errors)
            writer.writerow(row)

    lines = [
        "# v101 PDF 원문/이미지 추출 결과",
        "",
        f"- 입력 폴더: `{input_dir}`",
        f"- 출력 폴더: `{out_dir}`",
        f"- 처리 논문: {len(summaries)}편",
        f"- 렌더링 페이지: {sum(item.rendered_pages for item in summaries)}장",
        f"- embedded 이미지: {sum(item.embedded_images for item in summaries)}개",
        f"- image-block crop: {sum(item.image_crops for item in summaries)}개",
        "",
        "## 산출물",
        "",
        "- `texts/*.txt`: 페이지 구분자가 포함된 원문 텍스트",
        "- `json/*.json`: 페이지별 텍스트와 bbox, 인용 번호 메타데이터, 이미지 정보",
        "- 확인한 사설 문자 치환: U+E0D5 → -, U+E0C8 → =, U+E0C9 → ≡",
        "- 행·열 구조가 안정적으로 잡힌 표는 Markdown으로 정리하며, 불확실한 표는 확인 표시와 원본 crop을 남깁니다.",
        "- 안전하게 복원한 수식은 읽기 좋은 텍스트로 출력하고, 복잡하거나 애매한 수식은 검토 표시와 원본 crop을 남깁니다.",
        "- 그림 캡션과 벡터/이미지 영역을 이용해 짧은 축·범례 텍스트의 본문 혼입을 억제합니다. 원문 블록은 JSON에 보존합니다.",
        "- `quality_report.json`: 무결성 실패와 검수 경고를 별도로 표시합니다.",
        "- `comparison.html`: 원본 페이지와 TXT를 대조합니다.",
        "- 줄 끝 하이픈은 soft hyphen 또는 같은 PDF의 붙여 쓴 표기를 근거로 제거합니다. 불명확하면 하이픈을 유지하며, 문단 사이를 합치지 않습니다.",
        "- `pages/<paper_id>/page-###.png`: 그래프/표/벡터 요소까지 보존한 전체 페이지 이미지",
        "- `images/<paper_id>/*`: `figure3_p004_crop01.png` 형식의 그림 파일과 `image_p004_...` 형식의 미분류 이미지",
        "  `crop`은 페이지에서 잘라 렌더링한 이미지이고, `xref`는 PDF 내부 원본 이미지 객체의 ID입니다.",
        "",
        "## 논문별 요약",
        "",
        "| paper | pages | chars | blocks | rendered | embedded | crops | errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item.paper_id} | {item.pages} | {item.text_chars} | {item.text_blocks} | "
            f"{item.rendered_pages} | {item.embedded_images} | {item.image_crops} | {len(item.errors)} |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def comparable_text(value: str) -> str:
    # 원문과 결과를 비교할 때 허용할 표기 차이를 정리합니다. 저장할 본문을 만드는 함수는 아닙니다.
    # 공백·첨자·하이픈 차이를 무시하므로 이 비교만으로 정확한 표기 복원까지 보장할 수는 없습니다.
    value = re.sub(r"\[CITATION:([^\]]+)\]", r"\1", value)
    value = dehyphenate_wrapped_words(normalize_pdf_symbols(value))
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"\s+", "", value.replace("\x00", "").replace("\u00ad", ""))
    return re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", value)


def verify_output(output: Path) -> int:
    # 원본 블록 누락/중복, TXT·JSON 일치, 이미지 파일 읽기 등을 검사합니다.
    # failures는 무결성 검사 실패, warnings는 수식/표/그림 경계 등의 검토 필요 항목입니다.
    # 실패 0건이어도 읽기 순서나 수식 의미가 모두 정확하다는 뜻은 아닙니다.
    manifest = json.loads((output / "manifest.json").read_text())
    results = []
    sections = []
    for paper in manifest["papers"]:
        payload = json.loads((output / "json" / f"{paper['paper_id']}.json").read_text())
        failures = list(paper["errors"])
        warnings = []
        if not payload["pages"]:
            results.append(dict(paper=paper["paper_id"], pages=0,
                                failures=failures or ["No pages extracted"], warnings=[]))
            continue
        try:
            if sha256_file(Path(paper["source_pdf"])) != paper["source_sha256"]:
                failures.append("source hash mismatch")
            doc = fitz.open(paper["source_pdf"])
        except Exception as exc:
            results.append(dict(paper=paper["paper_id"], pages=len(payload["pages"]),
                                failures=failures + [f"source unreadable: {exc}"], warnings=[]))
            continue
        with doc:
            if len(doc) != len(payload["pages"]):
                failures.append("page count mismatch")
            for page, saved in zip(doc, payload["pages"]):
                native = {b[5]: b[4] for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()}
                used = []
                for block in saved["blocks"]:
                    ids = block.get("source_block_nos", [block["block_no"]])
                    used.extend(ids)
                    raw = "".join(native.get(n, "") for n in ids)
                    actual = block.get("native_text", block["text"])
                    if comparable_text(raw) != comparable_text(actual):
                        failures.append(f"page {page.number+1} block {ids}: source text mismatch")
                    if "region_image" in block and not (output / block["region_image"]).is_file():
                        failures.append(f"missing region: {block['region_image']}")
                if len(used) != len(set(used)):
                    failures.append(f"page {page.number+1}: duplicate blocks")
                for n in set(native) - set(used):
                    if not native[n].startswith("Downloaded from "):
                        failures.append(f"page {page.number+1}: missing block {n}")
                if not saved["rendered_png"] or not (output / saved["rendered_png"]).is_file():
                    failures.append(f"page {page.number+1}: missing render")
                expected = "\n\n".join(b["text"] for b in saved["blocks"] if not b.get("excluded_from_body"))
                if expected != saved["text"]:
                    failures.append(f"page {page.number+1}: page/block mismatch")
                warnings.extend(dict(w, page=page.number+1) for w in saved["warnings"])
                links = " ".join(f'<a href="{html.escape(b["region_image"])}">{html.escape(b["kind"])} source</a>'
                                 for b in saved["blocks"] if "region_image" in b)
                sections.append(f'<section><h2>{html.escape(paper["paper_id"])} / {page.number+1}</h2>{links}'
                                f'<div class="pair"><img loading="lazy" src="{html.escape(saved["rendered_png"] or "")}">'
                                f'<pre>{html.escape(saved["text"])}</pre></div></section>')
        for asset in payload["images"]:
            try:
                pix = fitz.Pixmap(str(output / asset["path"]))
                if (pix.width, pix.height) != (asset["width"], asset["height"]):
                    failures.append(f"image dimensions mismatch: {asset['path']}")
            except Exception as exc:
                failures.append(f"image unreadable: {asset['path']}: {exc}")
        expected_txt = "".join(f"\n\n===== PAGE {p['page']} =====\n\n{p['text']}" for p in payload["pages"]).strip() + "\n"
        if (output / "texts" / f"{paper['paper_id']}.txt").read_text() != expected_txt:
            failures.append("TXT/JSON mismatch")
        results.append(dict(paper=paper["paper_id"], pages=len(payload["pages"]), failures=failures, warnings=warnings))
    count = sum(len(r["failures"]) for r in results)
    warning_count = sum(len(r["warnings"]) for r in results)
    report = dict(workflow="v101_generic_pdf_extraction", failures=count, warnings=warning_count, papers=results,
                  scope="Source-block coverage and preservation, output consistency, and asset decoding. "
                        "Math/table reconstructions and figure boundaries require review; warnings are not a pass.")
    (output / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    style = '<style>body{font:15px system-ui;margin:24px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}section{border-top:1px solid #aaa;padding:20px 0}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>'
    (output / "comparison.html").write_text('<!doctype html><meta charset="utf-8"><title>PDF / TXT</title>' + style
        + f'<h1>PDF / TXT</h1><p>Integrity failures: {count}; review warnings: {warning_count}</p>'
        + '<a href="quality_report.json">Quality report</a>' + "\n".join(sections))
    print(f"integrity failures={count}; review warnings={warning_count}; report={output / 'quality_report.json'}")
    return count


def run_self_tests() -> None:
    # 작은 가상 입력으로 이전에 고친 문제가 재발하는지 확인합니다. 실제 논문 전수검사는 아닙니다.
    import unittest
    import tempfile
    from unittest.mock import patch

    def block(number, bbox, text="body"):
        return {"block_no": number, "bbox": bbox, "text": text}
    
    
    class ExtractionRegressionTests(unittest.TestCase):
        def test_single_column_keeps_centered_equation_in_place(self):
            with fitz.open() as doc:
                page = doc.new_page(width=600, height=800)
                items = [block(1, [50, 100, 200, 120]),
                         block(2, [260, 150, 380, 170], "x = y"),
                         block(3, [50, 200, 200, 220])]
                self.assertEqual([b["block_no"] for b in reading_order(items, page)], [1, 2, 3])

        def test_graph_axis_is_not_a_safe_top_boundary(self):
            with tempfile.TemporaryDirectory() as tmp, fitz.open() as doc:
                page = doc.new_page(width=600, height=800)
                page.draw_line((60, 100), (280, 100))
                page.draw_line((60, 300), (280, 300))
                page.draw_line((60, 100), (60, 300))
                page.draw_line((60, 250), (220, 140))
                page.insert_text((60, 340), "Figure 1. Test graph")
                blocks = text_blocks(page)
                warnings = []
                assets = save_figure_regions(page, figure_captions(blocks), blocks,
                                             Path(tmp), Path(tmp), 1, warnings)
                self.assertEqual(assets[0].kind, "figure_page_fallback")
                self.assertTrue(fitz.Rect(assets[0].bbox).contains(fitz.Rect(60, 100, 280, 300)))
                self.assertEqual(warnings[0]["code"], "figure_region_uncertain")

        def test_output_name_collision_rejected_before_writing(self):
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "output"
                for names in (("paper+a.pdf", "paper a.pdf"), ("Paper.pdf", "paper.pdf")):
                    with self.assertRaises(ValueError):
                        validate_output_names([Path(n) for n in names], output)
                self.assertFalse(output.exists())

        def test_existing_output_owner_is_protected(self):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "json").mkdir()
                (root / "json/paper.json").write_text(json.dumps(
                    dict(summary=dict(source_pdf=str(root / "original/paper.pdf")))))
                with self.assertRaises(ValueError):
                    validate_output_names([root / "other/paper.pdf"], root)

        def test_encrypted_paper_does_not_stop_following_paper(self):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with fitz.open() as doc:
                    doc.new_page().insert_text((60, 100), "Normal text for extraction.")
                    doc.save(root / "locked.pdf", encryption=fitz.PDF_ENCRYPT_AES_256,
                             owner_pw="owner", user_pw="secret")
                    doc.save(root / "normal.pdf")
                output = root / "output"
                summaries = [extract_pdf(root / name, output, .5, .5)
                             for name in ("locked.pdf", "normal.pdf")]
                self.assertTrue(summaries[0].errors)
                self.assertFalse(summaries[1].errors)
                self.assertIn("Normal text", (output / "texts/normal.txt").read_text())
                write_manifest(output, root, summaries)
                self.assertGreater(verify_output(output), 0)
                self.assertTrue((output / "quality_report.json").exists())

        def test_unexpected_extraction_exception_is_recorded(self):
            def fail(*args):
                raise RuntimeError("synthetic page failure")

            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with fitz.open() as doc:
                    doc.new_page()
                    doc.save(root / "paper.pdf")
                with patch.dict(extract_pdf.__globals__, _extract_pdf=fail):
                    result = extract_pdf(root / "paper.pdf", root / "out", 1, 1)
                self.assertIn("synthetic page failure", result.errors[0])

        def test_wrapped_word_uses_document_evidence(self):
            self.assertEqual(dehyphenate_wrapped_words("technolog-\nically", {"technologically"}),
                             "technologically")

        def test_no_evidence_or_conflict_preserves_hyphen(self):
            for forms in (None, {"technologically", "technolog-ically"}):
                self.assertEqual(dehyphenate_wrapped_words("technolog-\nically", forms),
                                 "technolog-ically")
            self.assertEqual(dehyphenate_wrapped_words("high-\naspect-ratio", {"high-aspect-ratio"}),
                             "high-aspect-ratio")

        def test_explicit_soft_hyphen(self):
            self.assertEqual(dehyphenate_wrapped_words("technolog\u00ad\nically"), "technologically")

        def test_no_join_across_paragraphs(self):
            self.assertEqual(dehyphenate_wrapped_words("some-\n\nthing", {"something"}), "some-\n\nthing")

        def test_document_evidence_excludes_wrapped_and_compound_tokens(self):
            with fitz.open() as doc:
                page = doc.new_page()
                page.insert_text((40, 50), "techno-\nlogically\nhigh-aspect-ratio\ntechnologically")
                forms = document_word_forms(doc)
                self.assertIn("technologically", forms)
                self.assertIn("high-aspect-ratio", forms)
                self.assertNotIn("techno", forms)
                self.assertNotIn("aspect", forms)

        def test_fraction_is_not_flattened(self):
            chars = [dict(c="1", size=10, origin=(20, 60), bbox=(20, 50, 25, 62)),
                     dict(c="x", size=10, origin=(20, 80), bbox=(20, 70, 25, 82))]
            self.assertIsNone(linear_equation(chars, [(18, 27, 66)]))

        def test_character_coordinates_keep_chemical_subscript(self):
            chars = [dict(c="H", size=10, origin=(20, 60), bbox=(20, 50, 27, 62)),
                     dict(c="2", size=7, origin=(27, 63), bbox=(27, 56, 31, 65))]
            self.assertEqual(linear_equation(chars, []), "H₂")

        def test_prose_with_arrow_is_not_an_equation(self):
            paragraph = block(1, [60, 70, 300, 300], "Resistivity at T → 0 K; r = 5.")
            self.assertFalse(math_candidate(paragraph))
    
        def test_left_column_before_right_even_if_left_starts_lower(self):
            with fitz.open() as doc:
                page = doc.new_page(width=625, height=818)
                left = block(1, [60, 400, 300, 760])
                right = block(2, [324, 70, 564, 760])
                self.assertEqual([b["block_no"] for b in reading_order([right, left], page)], [1, 2])
    
        def test_full_width_bottom_caption_follows_both_columns(self):
            with fitz.open() as doc:
                page = doc.new_page(width=625, height=818)
                items = [block(1, [60, 70, 300, 500]), block(2, [324, 70, 564, 500]),
                         block(3, [60, 700, 564, 740], "Figure 1. caption")]
                self.assertEqual([b["block_no"] for b in reading_order(items, page)], [1, 2, 3])
    
        def test_caption_below_graph_preferred_to_previous_caption(self):
            captions = [dict(figure="Figure 7", bbox=[60, 400, 300, 500]),
                        dict(figure="Figure 8", bbox=[60, 710, 300, 760])]
            self.assertEqual(match_figure(fitz.Rect(60, 520, 300, 690), captions)["figure"], "Figure 8")
    
        def test_body_reference_is_not_caption(self):
            self.assertEqual(figure_captions([block(1, [0, 0, 1, 1], "Discussion of\nFigure 3d. Results follow.")]), [])
            self.assertEqual(figure_captions([dict(block(1, [0, 0, 1, 1], "Figure 4. More discussion"), caption_candidate=False)]), [])
    
        def test_superscript_and_subscript_have_distinct_positions(self):
            line = {"spans": [
                {"text": "10", "size": 10, "origin": [0, 100], "color": 0},
                {"text": "5", "size": 7, "origin": [10, 96], "color": 0},
                {"text": " Bi", "size": 10, "origin": [14, 100], "color": 0},
                {"text": "2", "size": 7, "origin": [28, 103], "color": 0},
            ]}
            self.assertEqual([a["marker"] for a in script_spans(line, [])], ["⁵", "₂"])
    
        def test_reference_at_split_block_start(self):
            line = {"spans": [
                {"text": "8,9,17", "size": 7, "origin": [0, 100], "bbox": [0, 93, 20, 100], "color": 0x1b4ba0},
                {"text": " with thicknesses", "size": 10, "origin": [20, 100], "bbox": [20, 90, 100, 100], "color": 0},
            ]}
            self.assertEqual(citation_spans(line)[0]["marker"], "[CITATION:8,9,17]")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ExtractionRegressionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    # 실행 시작점입니다. --self-test는 테스트만, --verify-only는 기존 결과 검증만 실행합니다.
    # 기본 실행은 추출 → 요약 저장 → 검증 순서이며, --limit 10이면 처음 10편만 처리합니다.
    parser = argparse.ArgumentParser(description="v101 PDF text and asset extraction workflow")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--page-zoom", type=float, default=2.0)
    parser.add_argument("--crop-zoom", type=float, default=2.5)
    parser.add_argument("--verify-only", action="store_true", help="Check existing outputs without extraction")
    parser.add_argument("--self-test", action="store_true", help="Run built-in regression tests")
    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
        return
    if args.verify_only:
        raise SystemExit(bool(verify_output(args.output_dir)))
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.page_zoom <= 0 or args.crop_zoom <= 0:
        parser.error("zoom must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = find_pdfs(args.input_dir, args.limit)
    if not pdfs:
        raise SystemExit(f"No PDF files found in {args.input_dir}")
    try:
        validate_output_names(pdfs, args.output_dir)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))

    summaries: list[PaperSummary] = []
    for pdf_path in pdfs:
        summary = extract_pdf(pdf_path, args.output_dir, args.page_zoom, args.crop_zoom)
        summaries.append(summary)
        print(
            f"{summary.paper_id}: pages={summary.pages} chars={summary.text_chars} "
            f"embedded={summary.embedded_images} crops={summary.image_crops} errors={len(summary.errors)}",
            flush=True,
        )

    write_manifest(args.output_dir, args.input_dir, summaries)
    print(f"wrote {args.output_dir / 'manifest.json'}", flush=True)
    raise SystemExit(bool(verify_output(args.output_dir)))


if __name__ == "__main__":
    main()
