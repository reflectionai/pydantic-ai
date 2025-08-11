"""Utility for formatting JSON decode errors with visual context."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class JsonErrorInfo:
    """Extracted information from a JSONDecodeError."""

    msg: str
    pos: int
    lineno: int
    colno: int
    doc: str
    error_line: str
    lines: list[str]


def extract_json_error_info(error: json.JSONDecodeError) -> JsonErrorInfo:
    """Extract structured information from a JSONDecodeError.

    Args:
        error: The JSONDecodeError to extract from

    Returns:
        JsonErrorInfo with extracted data
    """
    doc = error.doc or ''
    lines = doc.splitlines()

    # Get the problematic line (lineno is 1-based)
    error_line = ''
    if error.lineno >= 1 and error.lineno <= len(lines):
        error_line = lines[error.lineno - 1]

    return JsonErrorInfo(
        msg=error.msg,
        pos=error.pos,
        lineno=error.lineno,
        colno=error.colno,
        doc=doc,
        error_line=error_line,
        lines=lines,
    )


def format_json_error_visual(error_info: JsonErrorInfo) -> str:
    """Format JsonErrorInfo with visual context similar to compiler errors.

    Args:
        error_info: The extracted error information

    Returns:
        A formatted string showing the error location with visual indicators
    """
    if not error_info.doc:
        return f'{error_info.msg} at position {error_info.pos}'

    # If we don't have valid line/col info, fall back to basic error
    if error_info.lineno < 1 or error_info.lineno > len(error_info.lines):
        return f'{error_info.msg} at position {error_info.pos}'

    # Create the visual indicator
    # colno is 1-based, so we need colno-1 spaces before the caret
    caret_pos = max(0, error_info.colno - 1)
    visual_indicator = ' ' * caret_pos + '^'

    # Build the formatted error message
    parts = [
        f'JSON parsing error, line {error_info.lineno}:',
        f'    {error_info.error_line}',
        f'    {visual_indicator}',
        f'JSONDecodeError: {error_info.msg}',
    ]

    return '\n'.join(parts)


def format_json_decode_error(error: json.JSONDecodeError) -> str:
    """Format a JSONDecodeError with visual context similar to compiler errors.

    Args:
        error: The JSONDecodeError to format

    Returns:
        A formatted string showing the error location with visual indicators
    """
    error_info = extract_json_error_info(error)
    return format_json_error_visual(error_info)


def create_json_error_context(error: json.JSONDecodeError, model_name: str, chunk_count: int) -> dict[str, Any]:
    """Create structured context for JSON decode errors.

    Args:
        error: The JSONDecodeError
        model_name: Name of the model that failed
        chunk_count: Number of chunks processed before failure

    Returns:
        Dictionary with structured error context
    """
    error_info = extract_json_error_info(error)
    formatted_error = format_json_error_visual(error_info)

    return {
        'model_name': model_name,
        'chunk_count': chunk_count,
        'json_error_msg': error_info.msg,
        'json_error_pos': error_info.pos,
        'json_error_lineno': error_info.lineno,
        'json_error_colno': error_info.colno,
        'formatted_error': formatted_error,
        'problematic_content_preview': error_info.doc[:500] + '...'
        if len(error_info.doc) > 500
        else error_info.doc or 'N/A',
    }
