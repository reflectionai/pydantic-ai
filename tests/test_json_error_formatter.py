"""Tests for JSON error formatting utilities."""

import json
from typing import Any

import pytest

from pydantic_ai._json_error_formatter import (
    JsonErrorInfo,
    create_json_error_context,
    extract_json_error_info,
    format_json_decode_error,
    format_json_error_visual,
)


@pytest.mark.parametrize(
    'test_case',
    [
        pytest.param(
            {
                'id': 'basic_unterminated_string',
                'json_input': '{"key": "value"',
                'is_mock': False,
                'expected_lineno': 1,
                'expected_error_line': '{"key": "value"',
                'expected_lines': ['{"key": "value"'],
            },
            id='basic_unterminated_string',
        ),
        pytest.param(
            {
                'id': 'multiline_missing_comma',
                'json_input': '{\n  "key": "value"\n  "missing_comma": true\n}',
                'is_mock': False,
                'expected_lineno': 3,
                'expected_error_line': '  "missing_comma": true',
                'expected_lines': ['{', '  "key": "value"', '  "missing_comma": true', '}'],
            },
            id='multiline_missing_comma',
        ),
        pytest.param(
            {
                'id': 'empty_document',
                'json_input': '',
                'is_mock': True,
                'mock_msg': 'test error',
                'mock_pos': 0,
                'expected_lineno': 1,
                'expected_error_line': '',
                'expected_lines': [],
            },
            id='empty_document',
        ),
    ],
)
def test_extract_json_error_info(test_case: dict[str, Any]):
    """Test extraction of JSON error information from various scenarios."""
    if test_case['is_mock']:
        # Create a mock JSONDecodeError
        error = json.JSONDecodeError(test_case['mock_msg'], test_case['json_input'], test_case['mock_pos'])
        expected = JsonErrorInfo(
            msg=test_case['mock_msg'],
            pos=test_case['mock_pos'],
            lineno=test_case['expected_lineno'],
            colno=1,
            doc=test_case['json_input'],
            error_line=test_case['expected_error_line'],
            lines=test_case['expected_lines'],
        )
    else:
        # Parse real JSON to get actual error
        try:
            json.loads(test_case['json_input'])
            pytest.fail('Expected JSONDecodeError but parsing succeeded')
        except json.JSONDecodeError as e:
            # For real JSON errors, some values vary by implementation
            expected = JsonErrorInfo(
                msg=e.msg,  # Use actual message
                pos=e.pos,  # Use actual position
                lineno=test_case['expected_lineno'],
                colno=e.colno,  # Use actual column
                doc=test_case['json_input'],
                error_line=test_case['expected_error_line'],
                lines=test_case['expected_lines'],
            )
            error = e

    error_info = extract_json_error_info(error)
    assert isinstance(error_info.msg, str)  # Always check message is string
    assert error_info == expected


@pytest.mark.parametrize(
    'test_case',
    [
        pytest.param(
            {
                'error_info': JsonErrorInfo(
                    msg='Expecting property name enclosed in double quotes',
                    pos=15,
                    lineno=1,
                    colno=16,
                    doc='{"key": "value",}',
                    error_line='{"key": "value",}',
                    lines=['{"key": "value",}'],
                ),
                'expected_contains': [
                    'JSON parsing error, line 1:',
                    '{"key": "value",}',
                    '^',
                    'JSONDecodeError: Expecting property name enclosed in double quotes',
                ],
                'expected_caret_line': '    ' + ' ' * 15 + '^',
                'fallback_format': None,
            },
            id='basic_visual_formatting',
        ),
        pytest.param(
            {
                'error_info': JsonErrorInfo(
                    msg='Invalid syntax',
                    pos=20,
                    lineno=2,
                    colno=5,
                    doc='{\n  "ke',
                    error_line='  "ke',
                    lines=['{', '  "ke'],
                ),
                'expected_contains': ['JSON parsing error, line 2:', '  "ke', '^'],
                'expected_caret_line': '    ' + ' ' * 4 + '^',
                'fallback_format': None,
            },
            id='multiline_visual_formatting',
        ),
        pytest.param(
            {
                'error_info': JsonErrorInfo(
                    msg='No JSON object could be decoded', pos=0, lineno=1, colno=1, doc='', error_line='', lines=[]
                ),
                'expected_contains': [],
                'expected_caret_line': None,
                'fallback_format': 'No JSON object could be decoded at position 0',
            },
            id='empty_document_fallback',
        ),
        pytest.param(
            {
                'error_info': JsonErrorInfo(
                    msg='Parse error',
                    pos=5,
                    lineno=10,  # Invalid line number
                    colno=1,
                    doc='short',
                    error_line='',
                    lines=['short'],
                ),
                'expected_contains': [],
                'expected_caret_line': None,
                'fallback_format': 'Parse error at position 5',
            },
            id='invalid_line_fallback',
        ),
    ],
)
def test_format_json_error_visual(test_case: dict[str, Any]):
    """Test visual formatting of JSON errors in various scenarios."""
    formatted = format_json_error_visual(test_case['error_info'])

    if test_case['fallback_format']:
        # Test fallback formatting
        assert formatted == test_case['fallback_format']
    else:
        # Test normal visual formatting
        for expected_text in test_case['expected_contains']:
            assert expected_text in formatted

        if test_case['expected_caret_line']:
            lines = formatted.split('\n')
            caret_line = next(line for line in lines if '^' in line)
            assert caret_line.rstrip() == test_case['expected_caret_line']


def test_format_json_decode_error_integration():
    """Test the high-level formatting function."""
    malformed_json = '{"test": invalid}'
    try:
        json.loads(malformed_json)
    except json.JSONDecodeError as e:
        formatted = format_json_decode_error(e)

        expected_contains = ['JSON parsing error, line 1:', '{"test": invalid}', '^', 'JSONDecodeError:']

        for expected_text in expected_contains:
            assert expected_text in formatted


def test_create_json_error_context():
    """Test creation of structured error context."""
    malformed_json = '{"key": "value"'
    try:
        json.loads(malformed_json)
    except json.JSONDecodeError as e:
        context = create_json_error_context(e, 'test-model', 5)

        expected_context = {
            'model_name': 'test-model',
            'chunk_count': 5,
            'json_error_msg': e.msg,  # Use actual message since it varies
            'json_error_pos': 15,
            'json_error_lineno': 1,
            'json_error_colno': 16,
            'formatted_error': context['formatted_error'],  # Use actual formatted error
            'problematic_content_preview': malformed_json,
        }

        assert context == expected_context
        # Additional checks for the formatted error content
        assert 'JSON parsing error, line 1:' in context['formatted_error']


def test_create_json_error_context_long_content():
    """Test context creation with long content gets truncated."""
    long_json = '{"key": "' + 'x' * 600 + '"'
    try:
        json.loads(long_json)
    except json.JSONDecodeError as e:
        context = create_json_error_context(e, 'test-model', 1)

        # Check that long content gets truncated
        preview = context['problematic_content_preview']
        expected_preview = {
            'length': 503,  # 500 + '...'
            'ends_with': '...',
            'starts_with': '{"key": "xxx',
        }

        assert len(preview) == expected_preview['length']
        assert preview.endswith(expected_preview['ends_with'])
        assert preview.startswith(expected_preview['starts_with'])


@pytest.mark.parametrize(
    'caret_position,expected_caret_line',
    [
        (1, '    ^'),  # First column: 4 spaces + caret
        (5, '        ^'),  # Fifth column: 4 spaces + 4 spaces + caret
        (10, '             ^'),  # Tenth column: 4 spaces + 9 spaces + caret
    ],
)
def test_caret_positioning(caret_position: int, expected_caret_line: str):
    """Test that caret is positioned correctly for different column positions."""
    error_info = JsonErrorInfo(
        msg='Test error', pos=0, lineno=1, colno=caret_position, doc='x' * 20, error_line='x' * 20, lines=['x' * 20]
    )

    formatted = format_json_error_visual(error_info)
    lines = formatted.split('\n')
    caret_line = next(line for line in lines if '^' in line)

    assert caret_line.rstrip() == expected_caret_line
