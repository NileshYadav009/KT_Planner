from typing import Any, Dict, List

# Below this many surviving columns a table stops being a table, so pruning
# never reduces one further than this even if the data is that sparse.
_MIN_KEPT_COLUMNS = 2


def prune_empty_columns(columns: List[str], rows: List[Dict[str, Any]]) -> List[str]:
    """Drop columns for which EVERY row is empty.

    A column no row has any value for carries no information, yet each such
    cell still renders the explicit "Not covered during KT" placeholder — on
    real generated KTs that produced tables where four of five columns were
    nothing but that phrase repeated, crowding the one column that did have
    content down to an unreadable width. Partially-filled columns are kept
    untouched, so a genuine per-row "not covered" signal still shows.

    Returns the columns to render; row dicts are left alone (renderers read
    cells by column name, so extra keys are simply not displayed).
    """
    kept = [
        col for col in columns
        if any(str((row or {}).get(col, "") or "").strip() for row in rows or [])
    ]
    if len(kept) < _MIN_KEPT_COLUMNS:
        return [col for col in columns if col][:max(_MIN_KEPT_COLUMNS, len(kept))]
    return kept


def build_block(title: str, columns: List[str], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    clean_rows = [row for row in rows if isinstance(row, dict)]
    return {
        "type": "DecisionTable",
        "title": title,
        "columns": prune_empty_columns([col for col in columns if col], clean_rows),
        "rows": clean_rows,
    }


def parse_table_rows(value: Any, columns: List[str]) -> List[Dict[str, str]]:
    """Best-effort split of a table-type field's `\\n`-joined string value
    (see field_populator.py's `type == "table"` extraction) into row dicts
    keyed by `columns`. Pipe-delimited lines map cell-by-cell; anything else
    falls back to putting the whole line in the first column, since natural
    transcript speech rarely comes pre-formatted as a real table.
    """
    text = str(value or "")
    rows: List[Dict[str, str]] = []
    for line in [line.strip() for line in text.split("\n") if line.strip()]:
        if "|" in line:
            cells = [cell.strip() for cell in line.split("|")]
            rows.append({columns[idx]: cells[idx] if idx < len(cells) else "" for idx in range(len(columns))})
        else:
            rows.append({columns[0]: line} if columns else {"Step": line})
    return rows
