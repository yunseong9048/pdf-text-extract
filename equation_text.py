"""Generic PDF equation helpers based on text geometry; no paper-specific answers."""
from __future__ import annotations

import re
from typing import Any

import fitz

def normalize_pdf_symbols(value: str) -> str:
    # 합자와 v101 파일럿에서 원본 대조한 세 결합 기호를 정규화합니다.
    # U+E0D5=단일 결합/대시, U+E0C8=이중 결합, U+E0C9=삼중 결합.
    value = value.translate(str.maketrans({
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
        "\ue0d5": "−", "\ue0c8": "=", "\ue0c9": "≡",
        "ð": "(", "Þ": ")",
    }))
    # Several embedded scientific fonts expose the equality and multiplication
    # signs through compatibility/control glyphs. Normalize only the contexts
    # where their mathematical role is clear; keep ambiguous standalone glyphs.
    value = re.sub(r"(?<=[A-Za-z0-9)])\s*¼", " = ", value)
    value = re.sub(r"(?<=\d)\x01(?=\s*10\b)", " × ", value)
    value = re.sub(r"\x02(?=CP\b)", "µ", value)
    value = re.sub(r"(\d\s*)\x02(?=M\b)", r"\1µ", value)
    # Some PDF font maps expose an in-token minus as form-feed (U+000C).
    return re.sub(r"(?<=[A-Za-z0-9)])\x0c(?=[A-Za-z0-9(])", "−", value)

def horizontal_rules(page: fitz.Page) -> list[tuple[float, float, float]]:
    # PDF의 선/얇은 사각형에서 수평선을 찾습니다. 분수선 및 그림 경계 추정에 사용합니다.
    rules = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                a, b = item[1:3]
                if abs(a.y - b.y) < .5 and abs(a.x - b.x) > 3:
                    rules.append((min(a.x, b.x), max(a.x, b.x), (a.y + b.y) / 2))
            elif item[0] == "re":
                rect = item[1]
                if rect.width > 3 and rect.height < 1.5:
                    rules.append((rect.x0, rect.x1, (rect.y0 + rect.y1) / 2))
    return rules

def raw_characters(page: fitz.Page) -> dict[int, list[dict[str, Any]]]:
    # 수식은 텍스트 저장 순서가 뒤섞일 수 있어, 글자 하나씩의 위치와 크기를 가져옵니다.
    result = {}
    for block in page.get_text("rawdict")["blocks"]:
        if block["type"] == 0:
            result[block["number"]] = [
                dict(c, c=normalize_pdf_symbols(c["c"]), size=span["size"])
                for line in block["lines"] for span in line["spans"]
                for c in span["chars"] if c["c"].strip()
            ]
    return result

def linear_equation(chars: list[dict[str, Any]], rules: list[tuple[float, float, float]]) -> str | None:
    """Only serialize one mathematical baseline; fractions need visual review."""
    # 한 줄 수식만 복원합니다. 알 수 없는 글자나 여러 기준선이 있으면 None을 반환합니다.
    # None은 '수식 없음'이 아니라 '자동 복원하지 않고 원본 이미지로 확인'이라는 뜻입니다.
    if not chars or any(0xe000 <= ord(c["c"]) <= 0xf8ff for c in chars):
        return None
    size = max(c["size"] for c in chars)
    base = [c for c in chars if c["size"] >= size * .85]
    origins = sorted(c["origin"][1] for c in base)
    if origins[-1] - origins[0] > size * .5:
        return None
    baseline = origins[len(origins) // 2]
    # 수평선 위아래에 글자가 모두 있으면 분수일 수 있으므로 한 줄로 펴지 않습니다.
    for x0, x1, y in rules:
        above = [c for c in chars if x0 - 1 <= (c["bbox"][0] + c["bbox"][2]) / 2 <= x1 + 1
                 and c["bbox"][3] <= y + 2]
        below = [c for c in chars if x0 - 1 <= (c["bbox"][0] + c["bbox"][2]) / 2 <= x1 + 1
                 and c["bbox"][1] >= y - 2]
        if above and below:
            return None

    groups: list[dict[str, Any]] = []
    # 큰 글자의 기준선에서 작게 위/아래로 벗어난 글자를 위첨자/아래첨자로 분류합니다.
    for char in sorted(chars, key=lambda c: c["bbox"][0]):
        dy = char["origin"][1] - baseline
        mode = ("sub" if dy > 1 else "sup") if char["size"] < size * .85 and abs(dy) > 1 else "base"
        gap = char["bbox"][0] - groups[-1]["x1"] if groups else 0
        if groups and groups[-1]["mode"] == mode and gap < size * .12:
            groups[-1]["text"] += char["c"]
            groups[-1]["x1"] = char["bbox"][2]
        else:
            groups.append(dict(mode=mode, text=char["c"], x1=char["bbox"][2], gap=gap))
    parts = []
    subs = str.maketrans("0123456789+-−()", "₀₁₂₃₄₅₆₇₈₉₊₋₋₍₎")
    sups = str.maketrans("0123456789+-−()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁽⁾")
    for group in groups:
        value = group["text"]
        if group["mode"] != "base":
            if re.fullmatch(r"[0-9+−()\-]+", value):
                value = value.translate(subs if group["mode"] == "sub" else sups)
            else:
                value = ("_" if group["mode"] == "sub" else "^") + "{" + value + "}"
        elif group["gap"] > size * .12 and parts:
            parts.append(" ")
        parts.append(value)
    value = re.sub(r"\s*([=+→↔⇌])\s*", r" \1 ", "".join(parts)).strip()
    value = normalize_pdf_symbols(value)
    # Unknown control/private glyphs in an equation usually indicate a lost
    # symbol. Prefer the saved crop over presenting that text as reconstructed.
    if any(ord(char) < 32 or 0xE000 <= ord(char) <= 0xF8FF for char in value):
        return None
    # A numeric coefficient followed by the ambiguous token "pf" commonly
    # results from a PDF font losing the distinction between π and p. Keep the
    # original crop available instead of guessing the intended notation.
    if re.search(r"\b\d+pf\b", value):
        return None
    return value

def math_candidate(block: dict[str, Any]) -> bool:
    # '='나 화살표가 있는 본문을 수식으로 오인하지 않도록 크기와 문장 길이를 제한합니다.
    # 이 조건은 후보 필터일 뿐, 수식임을 확정하는 판정은 아닙니다.
    text = block["text"]
    box = fitz.Rect(block["bbox"])
    words = re.findall(r"[A-Za-z]{3,}", text)
    prose_line = any(len(re.findall(r"[A-Za-z]{3,}", line)) >= 5 for line in text.splitlines())
    math_marks = re.search(r"[=→↔⇌≈≠≤≥±×÷∑√∞¼]", text)
    encoded_math = len(text) < 180 and any(mark in text for mark in "þðÞ\x02\x03\x06\x07\x04\x05")
    # Superscripts and compact symbol-heavy blocks often lose their operators
    # in PDF text extraction. Admit these as review candidates when they are
    # short and spatially compact, while continuing to exclude prose sentences.
    symbol_ratio = sum(not ch.isalnum() and not ch.isspace() for ch in text) / max(1, len(text))
    compact_math = len(text) < 120 and symbol_ratio >= .12 and len(words) <= 15
    return (box.height < 65 and len(text) < 350 and len(words) <= 15 and not prose_line
            and not re.match(r"(?i)^(where|Figure|Fig\.|Table|Scheme|DOI|https|Received)\b", text)
            and not re.search(r"[A-Za-z]{19,}", text)
            and (math_marks or encoded_math or compact_math or len(words) <= 2))



def process_equation_blocks(blocks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mark likely equation blocks while retaining their original text unchanged."""
    warnings = []
    for block in blocks:
        if math_candidate(block):
            block["equation_candidate"] = True
            warnings.append({"type": "equation_candidate", "block": block.get("block_no"), "text": block.get("text", "")})
    return blocks, warnings


def reconstruct_equation(chars: list[dict[str, Any]], rules: list[tuple[float, float, float]]) -> str | None:
    """Reconstruct safe baseline equations; leave fractions and ambiguous layouts for review."""
    if not chars:
        return None
    single_line = linear_equation(chars, rules)
    if single_line:
        return single_line
    # A rule with glyphs above and below signals a stacked fraction or complex layout.
    for x0, x1, y in rules:
        centers = [(c["bbox"][0] + c["bbox"][2]) / 2 for c in chars]
        above = any(x0 - 2 <= x <= x1 + 2 and c["bbox"][3] <= y + 2 for c, x in zip(chars, centers))
        below = any(x0 - 2 <= x <= x1 + 2 and c["bbox"][1] >= y - 2 for c, x in zip(chars, centers))
        if above and below:
            return None
    size = max(float(char.get("size", 0)) for char in chars)
    tolerance = max(1.5, size * .42)
    rows: list[list[dict[str, Any]]] = []
    baselines: list[float] = []
    for char in sorted(chars, key=lambda item: (item["origin"][1], item["bbox"][0])):
        baseline = float(char["origin"][1])
        nearest = min(range(len(baselines)), key=lambda i: abs(baselines[i] - baseline), default=None)
        if nearest is None or abs(baselines[nearest] - baseline) > tolerance:
            rows.append([char])
            baselines.append(baseline)
        else:
            rows[nearest].append(char)
            baselines[nearest] = sum(float(c["origin"][1]) for c in rows[nearest]) / len(rows[nearest])
    if len(rows) < 2 or len(rows) > 4:
        return None
    rendered = [linear_equation(row, []) for row in rows]
    if any(not row for row in rendered):
        return None
    if any(len(row.strip()) < 8 for row in rendered):
        return None
    return "\n".join(rendered)
