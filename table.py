"""Generic table caption detection, extraction, and Markdown formatting."""
from __future__ import annotations

import re
from typing import Any

import fitz
from equation_text import normalize_pdf_symbols

_TABLE_CAPTION = re.compile(r"(?i)^\s*Table\s+((?:S\s*)?\d+|[IVXLCDM]+)")
_NUMERIC = re.compile(r"^[\s+−\-]?(?:\d[\d.,:/%±×⁻⁺()°–—-]*|[<>≤≥∼~])(?:\s*(?:eV|nm|s|min|h|°C|K|%|at\.?\s*%|cm[²³⁻⁺/]+|mbar|Pa|μM|M|A|Ω(?:·cm)?))?\s*$", re.I)


def table_caption_match(text: str) -> re.Match[str] | None:
    """Match conventional and punctuation-free numbered table captions."""
    value = text.strip()
    match = _TABLE_CAPTION.match(value)
    if not match:
        return None
    remainder = value[match.end():]
    if not remainder or remainder[0] in ".:":
        return match
    # Some publishers omit punctuation between the table number and title.
    # Accept title case, while rejecting body sentences such as "Table 1 compares…".
    title = remainder.strip()
    return match if title and title[0].isupper() else None


def process_tables(blocks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tag likely table captions without changing their source text."""
    warnings = []
    for block in blocks:
        match = table_caption_match(str(block.get("text", "")))
        if match:
            block["caption_candidate"] = True
            warnings.append({"type": "table_candidate", "block": block.get("block_no"), "table_number": match.group(1).replace(" ", "")})
    return blocks, warnings


def _word_rows(page: fitz.Page, box: fitz.Rect) -> list[list[dict[str, Any]]]:
    words = []
    for item in page.get_text("words", clip=box, sort=True):
        x0, y0, x1, y1, text = item[:5]
        text = _clean_cell(text)
        if text:
            words.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text})
    rows: list[list[dict[str, Any]]] = []
    row_y: list[float] = []
    for word in sorted(words, key=lambda w: ((w["y0"] + w["y1"]) / 2, w["x0"])):
        center = (word["y0"] + word["y1"]) / 2
        index = min(range(len(row_y)), key=lambda i: abs(row_y[i] - center), default=None)
        if index is None or abs(row_y[index] - center) > max(2.0, (word["y1"] - word["y0"]) * .42):
            rows.append([word])
            row_y.append(center)
        else:
            rows[index].append(word)
            row_y[index] = sum((w["y0"] + w["y1"]) / 2 for w in rows[index]) / len(rows[index])
    return [sorted(row, key=lambda w: w["x0"]) for row in rows]


def _cells_for_row(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    gap_limit = max(7.0, sum(w["y1"] - w["y0"] for w in words) / max(len(words), 1) * .88)
    for word in words:
        if cells and word["x0"] - cells[-1]["x1"] <= gap_limit:
            cells[-1]["text"] += " " + word["text"]
            cells[-1]["x1"] = word["x1"]
            cells[-1]["y1"] = max(cells[-1]["y1"], word["y1"])
        else:
            cells.append(dict(word))
    return cells


def _is_numeric_cell(text: str) -> bool:
    value = text.strip().replace("−", "-").replace("±", " ")
    return bool(re.fullmatch(r"[+−-]?(?:\d[\d.,eE×*^/()\s<>~°–-]*|[<>≤≥∼~])(?:\s*(?:%|°C|°|K|eV|nm|s|min|h|mbar|Pa|S/cm|μM|M|A|cm[²³⁻⁺/]+|Ω(?:·cm)?))?", value, re.I))


def _is_temperature_label(text: str) -> bool:
    return bool(re.fullmatch(r"\s*\d+(?:\.\d+)?\s*(?:°|\x01)?\s*C\s*", text, re.I))


def _geometry_table(page: fitz.Page, box: fitz.Rect) -> tuple[list[list[str]] | None, float, list[str]]:
    line_rows = _word_rows(page, box)
    cell_rows = [_cells_for_row(row) for row in line_rows]
    data_indexes = []
    for index, cells in enumerate(cell_rows):
        line_text = " ".join(w["text"] for w in line_rows[index]).strip()
        if table_caption_match(line_text):
            continue
        numeric = sum(_is_numeric_cell(c["text"]) for c in cells)
        # Two-column tables are common (e.g. temperature vs. measured value).
        # Require at least two numeric cells so prose and caption years do not qualify.
        has_temperature = any(_is_temperature_label(c["text"]) for c in cells)
        if (len(cells) >= 2 and numeric >= max(2, round(len(cells) * .4))) or (
                len(cells) >= 2 and numeric >= 1 and has_temperature):
            data_indexes.append(index)
    # Some valid rows contain one textual measurement such as "Not reported"
    # and only one numeric value. Keep matching trailing rows when they share
    # the established table row shape.
    if data_indexes:
        row_sizes = {size: sum(len(cell_rows[i]) == size for i in data_indexes)
                     for size in {len(cell_rows[i]) for i in data_indexes}}
        common_size = max(row_sizes, key=lambda size: (row_sizes[size], size))
        for index in range(max(data_indexes) + 1, len(cell_rows)):
            if len(cell_rows[index]) == common_size and any(_is_numeric_cell(c["text"]) for c in cell_rows[index]):
                data_indexes.append(index)
            else:
                break
    if len(data_indexes) < 2:
        return None, 0.0, []
    counts: dict[int, int] = {}
    for index in data_indexes:
        counts[len(cell_rows[index])] = counts.get(len(cell_rows[index]), 0) + 1
    column_count = max(counts, key=lambda count: (counts[count], count))
    if column_count < 2 or column_count > 24:
        return None, 0.0, []
    aligned_rows = [cell_rows[i] for i in data_indexes if len(cell_rows[i]) == column_count]
    if len(aligned_rows) < 2:
        return None, 0.0, []
    centers = [sum((row[col]["x0"] + row[col]["x1"]) / 2 for row in aligned_rows) / len(aligned_rows)
               for col in range(column_count)]
    if any(b - a < 8 for a, b in zip(centers, centers[1:])):
        return None, 0.0, []
    first_data = data_indexes[0]
    header_rows = [cells for cells, words in zip(cell_rows[:first_data], line_rows[:first_data])
                   if not table_caption_match(" ".join(w["text"] for w in words))]
    parents: list[list[str]] = [[] for _ in centers]
    ordinary_headers: list[list[str]] = [[] for _ in centers]
    typical_gap = sorted(b - a for a, b in zip(centers, centers[1:]))
    median_gap = typical_gap[len(typical_gap) // 2] if typical_gap else 0
    for cells in header_rows:
        group_cells = ([cell for cell in cells
                        if not _is_numeric_cell(cell["text"])
                        and cell["x1"] - cell["x0"] >= median_gap * 1.15]
                       if len(cells) < column_count else [])
        for cell in cells:
            text = cell["text"].strip()
            if not text:
                continue
            if re.fullmatch(r"(?i)p\s*[<=>≤≥]\s*[\d.]+[).]*", text):
                continue
            parent_margin = max(8.0, median_gap * .12)
            covered = [i for i, center in enumerate(centers)
                       if cell["x0"] - parent_margin <= center <= cell["x1"] + parent_margin]
            nearest = min(range(len(centers)), key=lambda i: abs(centers[i] - (cell["x0"] + cell["x1"]) / 2))
            # Group headings are often centered over several columns but their
            # printed word does not physically span those columns. Detect a
            # centered label over a short, regular run of numeric columns.
            if (len(covered) < 2 and cell["x1"] - cell["x0"] >= median_gap * 1.15
                    and 0 < nearest < len(centers) - 1
                    and abs((centers[nearest] - centers[nearest - 1]) -
                            (centers[nearest + 1] - centers[nearest])) <= median_gap * .35):
                covered = [nearest - 1, nearest, nearest + 1]
            # A spanning title can be visually short because it is left aligned
            # inside a merged cell. Adjacent group titles provide the column-group
            # boundaries; if there is no next group, extend to the final column.
            if cell in group_cells:
                group_mid = (cell["x0"] + cell["x1"]) / 2
                nearest_group_column = min(range(len(centers)), key=lambda i: abs(centers[i] - group_mid))
                if len(group_cells) >= 2:
                    # Repeated group titles (e.g. two temperatures) generally
                    # each span a short, equal-sized run of child columns.
                    covered = [i for i in range(max(0, nearest_group_column - 1),
                                                min(len(centers), nearest_group_column + 2))]
                else:
                    # A single parent title normally spans the remaining child
                    # columns after the preceding row-label columns.
                    first_child = next((i for i, center in enumerate(centers)
                                        if center >= cell["x0"] - parent_margin),
                                       nearest_group_column)
                    covered = list(range(first_child, len(centers)))
            if len(covered) >= 2 and (cell["x1"] - cell["x0"] >= median_gap * .75):
                for i in covered:
                    if text not in parents[i]:
                        parents[i].append(text)
            else:
                ordinary_headers[nearest].append(text)
    headers = []
    for i in range(column_count):
        parts = parents[i] + ordinary_headers[i]
        headers.append(_format_chemical_subscripts(" ".join(dict.fromkeys(parts))))
    if not any(headers):
        return None, 0.0, []
    result = [headers]
    for index in data_indexes:
        cells = cell_rows[index]
        numeric_count = sum(_is_numeric_cell(c["text"]) for c in cells)
        meaningful_label = (any(re.search(r"[A-Za-z]{2,}", c["text"])
                                for c in cells if not _is_numeric_cell(c["text"]))
                            or (bool(cells) and not _is_numeric_cell(cells[0]["text"])
                                and _is_temperature_label(cells[0]["text"])))
        # Ignore isolated numeric labels below a table (often figure ticks or
        # repeated temperature marks), while retaining sparse rows with a label.
        if numeric_count < column_count and numeric_count < 4 and not meaningful_label:
            continue
        row = [""] * column_count
        for cell in cells:
            nearest = min(range(column_count), key=lambda i: abs(centers[i] - (cell["x0"] + cell["x1"]) / 2))
            row[nearest] = (row[nearest] + " " + cell["text"]).strip()
        result.append([_format_chemical_subscripts(cell) for cell in row])
    aligned_ratio = len(aligned_rows) / len(data_indexes)
    numeric_ratio = sum(_is_numeric_cell(c["text"]) for index in data_indexes for c in cell_rows[index]) / sum(len(cell_rows[i]) for i in data_indexes)
    confidence = round(.55 * aligned_ratio + .45 * numeric_ratio, 3)
    last_data = max(i for i in data_indexes if i < len(line_rows))
    notes = []
    last_y = sum((w["y0"] + w["y1"]) / 2 for w in line_rows[last_data]) / max(1, len(line_rows[last_data]))
    note_started = False
    previous_y = last_y
    for row_index in range(last_data + 1, len(cell_rows)):
        cells = cell_rows[row_index]
        text = " ".join(cell["text"] for cell in cells).strip()
        row_y = sum((w["y0"] + w["y1"]) / 2 for w in line_rows[row_index]) / max(1, len(line_rows[row_index]))
        marker_note = len(text) >= 12 and bool(
            re.match(r"^(?:[a-z*†‡]\s*(?=[A-Z])|notes?\b|\*\s*)", text))
        if not note_started and marker_note and row_y - last_y <= 24:
            note_started = True
        if note_started and row_y - previous_y <= 16 and text:
            notes.append(_format_chemical_subscripts(text))
            previous_y = row_y
        elif note_started:
            break
    return result, confidence, notes


def _single_data_row_table(page: fitz.Page, box: fitz.Rect) -> tuple[list[list[str]] | None, float, list[str]]:
    """Recover compact tables that contain one data row under a multi-column header."""
    rows = _word_rows(page, box)
    candidates = []
    for index, words in enumerate(rows):
        cells = _cells_for_row(words)
        numeric = sum(_is_numeric_cell(cell["text"]) for cell in cells)
        if numeric >= 4 and len(cells) >= 5:
            candidates.append((index, cells, numeric))
    if not candidates:
        return None, 0.0, []
    index, data_cells, _ = max(candidates, key=lambda item: (item[2], len(item[1])))
    centers = [(cell["x0"] + cell["x1"]) / 2 for cell in data_cells]
    headers = [""] * len(centers)
    for words in rows[:index]:
        line = " ".join(w["text"] for w in words)
        if table_caption_match(line):
            continue
        for cell in _cells_for_row(words):
            ci = min(range(len(centers)), key=lambda i: abs(centers[i] - (cell["x0"] + cell["x1"]) / 2))
            headers[ci] = (headers[ci] + " " + cell["text"]).strip()
    data = [_format_chemical_subscripts(_clean_cell(cell["text"])) for cell in data_cells]
    headers = [_format_chemical_subscripts(_clean_cell(cell)) for cell in headers]
    return [headers, data], .72, []


def _sparse_header_table(page: fitz.Page, box: fitz.Rect) -> tuple[list[list[str]] | None, float, list[str]]:
    """Read tables whose descriptive row labels sit beside regularly aligned numeric columns.

    Publisher tables often use spanning headers, leaving most cells in the header
    rows empty. Cell-count based reconstruction fragments those rows, so this path
    anchors columns on repeated numeric x positions and keeps each row label whole.
    """
    rows = _word_rows(page, box)
    numeric_rows: list[tuple[int, list[dict[str, Any]]]] = []
    for ri, words in enumerate(rows):
        vals = [w for w in words if _is_numeric_cell(w["text"])]
        label_before_values = any(w["x1"] < min(v["x0"] for v in vals) - 3
                                  and not _is_numeric_cell(w["text"]) for w in words) if vals else False
        if len(vals) >= 3 and (label_before_values or len(vals) >= 5):
            numeric_rows.append((ri, vals))
    if len(numeric_rows) < 2:
        return None, 0.0, []

    # Cluster numeric word centers across rows, so each data column has a stable x anchor.
    anchors: list[float] = []
    for _, vals in numeric_rows:
        for w in vals:
            x = (w["x0"] + w["x1"]) / 2
            nearest = min(range(len(anchors)), key=lambda i: abs(anchors[i] - x), default=None)
            if nearest is None or abs(anchors[nearest] - x) > 14:
                anchors.append(x)
            else:
                anchors[nearest] = (anchors[nearest] + x) / 2
    anchors.sort()
    # Keep only columns represented in at least two numerical rows.
    counts = [sum(1 for _, vals in numeric_rows if any(abs((w["x0"] + w["x1"]) / 2 - x) <= 14 for w in vals)) for x in anchors]
    anchors = [x for x, n in zip(anchors, counts) if n >= 2]
    if not 3 <= len(anchors) <= 20:
        return None, 0.0, []

    data_indices = [(ri, vals) for ri, vals in numeric_rows
                    if sum(any(abs((w["x0"] + w["x1"]) / 2 - x) <= 14 for w in vals) for x in anchors) >= 3]
    if len(data_indices) < 2:
        return None, 0.0, []
    first_data = min(ri for ri, _ in data_indices)
    output = [[""] * (len(anchors) + 1)]
    for ri, vals in data_indices:
        line = rows[ri]
        first_x = min(w["x0"] for w in vals)
        label = " ".join(w["text"] for w in line if w["x1"] < first_x - 3)
        record = [label] + [""] * len(anchors)
        for word in vals:
            x = (word["x0"] + word["x1"]) / 2
            ci = min(range(len(anchors)), key=lambda i: abs(anchors[i] - x))
            record[ci + 1] = word["text"]
        output.append(record)

    # Use the closest preceding header text for each column; this preserves sparse,
    # grouped headings without copying a spanning parent into unrelated cells.
    header_numbers: list[tuple[float, str]] = []
    parent_header_parts: list[str] = []
    for line in rows[:first_data]:
        line_nums = [w for w in line if _is_numeric_cell(w["text"])]
        header_numbers.extend(((w["x0"] + w["x1"]) / 2, w["text"]) for w in line_nums)
        # Join split words such as "temperature" + "(°C)" before deciding
        # whether the label spans the full group of numeric columns.
        words_only = [w for w in line if not _is_numeric_cell(w["text"])]
        grouped = _cells_for_row(words_only) if words_only else []
        for cell in grouped:
            txt = cell["text"].strip()
            mid = (cell["x0"] + cell["x1"]) / 2
            if txt and anchors[0] <= mid <= anchors[-1] and cell["x1"] - cell["x0"] > 18:
                parent_header_parts.append(txt)
    # Header values such as 25, 40, 60, 80 mark groups. Assign each value
    # to the numeric columns within its midpoint boundaries.
    header_numbers.sort()
    unique_headers: list[tuple[float, str]] = []
    for x, label in header_numbers:
        if unique_headers and abs(unique_headers[-1][0] - x) < 16:
            continue
        unique_headers.append((x, label))
    for ci, anchor in enumerate(anchors, start=1):
        if unique_headers:
            # A centered group header belongs to all columns between its
            # midpoint boundaries, not just the single nearest column.
            nearest = min(range(len(unique_headers)), key=lambda i: abs(unique_headers[i][0] - anchor))
            label = unique_headers[nearest][1]
            for hi, (hx, hlabel) in enumerate(unique_headers):
                left = (unique_headers[hi - 1][0] + hx) / 2 if hi else float("-inf")
                right = (hx + unique_headers[hi + 1][0]) / 2 if hi + 1 < len(unique_headers) else float("inf")
                if left <= anchor <= right:
                    label = hlabel
                    break
            output[0][ci] = label + (" °C" if re.fullmatch(r"\d+", label) else "")
        elif parent_header_parts:
            output[0][ci] = " ".join(dict.fromkeys(parent_header_parts))
    # A Markdown table cannot express merged cells. Keep the grouped numeric
    # headings repeated by column and avoid duplicating a spanning parent label.
    if parent_header_parts and not any(output[0][1:]):
        parent = " ".join(dict.fromkeys(re.sub(r"\s+", " ", p).strip() for p in parent_header_parts))
        output[0][1:] = [parent] * len(anchors)
    output[0][0] = output[0][0] or "Measurement"

    notes = []
    last_data = max(ri for ri, _ in data_indices)
    for line in rows[last_data + 1:]:
        note = " ".join(w["text"] for w in line).strip()
        if len(re.findall(r"[A-Za-z]{2,}", note)) >= 4:
            notes.append(_format_chemical_subscripts(note))
    output = [[_format_chemical_subscripts(_clean_cell(c)) for c in row] for row in output]
    return output, .82, notes


def extract_table_rows(page: fitz.Page, box: fitz.Rect) -> tuple[list[list[str]] | None, float, str | None, list[str]]:
    """Infer aligned rows from positioned words; use grid detection only as a cautious fallback."""
    try:
        rows, confidence, notes = _geometry_table(page, box)
        if rows and (confidence >= .58 or (confidence >= .50 and len(rows) >= 5)):
            return rows, confidence, None, notes
        rows, confidence, notes = _single_data_row_table(page, box)
        if rows:
            return rows, confidence, None, notes
        rows, confidence, notes = _sparse_header_table(page, box)
        if rows and confidence >= .58:
            return rows, confidence, None, notes
        found = page.find_tables(clip=box, strategy="text", min_words_vertical=2)
        candidates = [t for t in found.tables if t.row_count >= 2 and 2 <= t.col_count <= 24]
        if candidates:
            table = max(candidates, key=lambda t: t.row_count * t.col_count)
            fallback = [[_clean_cell(cell) for cell in row] for row in table.extract()]
            fallback = [row for row in fallback if any(row)]
            if len(fallback) >= 2:
                columns = max(map(len, fallback))
                padded = [row + [""] * (columns - len(row)) for row in fallback]
                fill = sum(bool(c) for row in padded for c in row) / (len(padded) * columns)
                fragments = sum(bool(re.fullmatch(r"[A-Za-z]", c.strip())) for row in padded[:3] for c in row)
                fragment_ratio = fragments / max(1, sum(bool(c.strip()) for row in padded[:3] for c in row))
                fallback_confidence = .55 * fill + .45 * (1 - fragment_ratio)
                if fallback_confidence >= .82:
                    return padded, round(fallback_confidence, 3), None, []
        return None, confidence, None, notes
    except Exception as exc:  # PyMuPDF text geometry varies across publishers.
        return None, 0.0, f"{type(exc).__name__}: {exc}", []


def _format_chemical_subscripts(text: str) -> str:
    sub = str.maketrans("0123456789+-", "₀₁₂₃₄₅₆₇₈₉₊₋")
    return re.sub(r"([A-Z][a-z]?|\))([0-9]+)", lambda m: m.group(1) + m.group(2).translate(sub), text)

def _clean_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    text = normalize_pdf_symbols(text).replace("\x00", " ").replace("\r", "\n")
    text = text.replace("\x02M", "μM")
    text = text.replace("◦", "°")
    text = re.sub(r"(?<=\d)\s+C\b", " °C", text)
    text = re.sub(r"\[\s*C\]", "[°C]", text)
    text = re.sub(r"(?<=\d)\s*\x01\s*C\b", " °C", text)
    text = "<br>".join(re.sub(r"\s+", " ", line).strip() for line in text.split("\n"))
    return text.replace("|", "\\|").strip()


def rows_to_markdown(rows: list[list[str]]) -> str:
    """Render extracted cells as a readable Markdown table with numeric alignment."""
    if not rows:
        return ""
    columns = max(map(len, rows))
    normalized = [list(row) + [""] * (columns - len(row)) for row in rows]
    header = normalized[0]
    body = normalized[1:]
    separators = []
    for column in range(columns):
        values = [row[column].strip() for row in body if row[column].strip()]
        numeric = bool(values) and sum(bool(_NUMERIC.fullmatch(re.sub(r"\s+", "", v))) for v in values) / len(values) >= 0.7
        separators.append("---:" if numeric else "---")
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(separators) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)
