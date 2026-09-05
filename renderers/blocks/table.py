from typing import Any, Dict, List


def build_block(title: str, columns: List[str], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "type": "DecisionTable",
        "title": title,
        "columns": [col for col in columns if col],
        "rows": [row for row in rows if isinstance(row, dict)],
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
