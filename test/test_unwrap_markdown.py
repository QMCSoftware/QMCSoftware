from scripts.unwrap_markdown import unwrap_markdown_text


def test_unwraps_plain_paragraph():
    source = (
        "Entries are grouped by publication year. Preprints are grouped by\n"
        "their first-submission year.\n"
    )

    assert unwrap_markdown_text(source) == (
        "Entries are grouped by publication year. Preprints are grouped by "
        "their first-submission year.\n"
    )


def test_unwraps_each_list_item_without_merging_adjacent_items():
    source = (
        '- First Author. "First title\n'
        '  continued." *Journal*, 2026.\n'
        '- Second Author. "Second title\n'
        '  continued." *Journal*, 2025.\n'
        "\n"
        "### Earlier publications\n"
    )

    assert unwrap_markdown_text(source) == (
        '- First Author. "First title continued." *Journal*, 2026.\n'
        '- Second Author. "Second title continued." *Journal*, 2025.\n'
        "\n"
        "### Earlier publications\n"
    )


def test_unwraps_ordered_and_nested_list_items():
    source = (
        "1. Parent item\n"
        "   continued.\n"
        "   - Nested item\n"
        "     continued.\n"
        "2. Next item.\n"
    )

    assert unwrap_markdown_text(source) == (
        "1. Parent item continued.\n"
        "   - Nested item continued.\n"
        "2. Next item.\n"
    )


def test_parent_continuation_after_nested_item_is_not_joined_to_child():
    source = (
        "- Parent item.\n"
        "  - Child item\n"
        "    continued.\n"
        "  Parent continuation.\n"
        "- Next item.\n"
    )

    assert unwrap_markdown_text(source) == (
        "- Parent item.\n"
        "  - Child item continued.\n"
        "  Parent continuation.\n"
        "- Next item.\n"
    )


def test_year_at_start_of_continuation_is_not_a_nested_ordered_list():
    source = (
        '- Author. "Title." *arXiv:2506.05391*,\n'
        "  2025. [arXiv](https://arxiv.org/abs/2506.05391).\n"
    )

    assert unwrap_markdown_text(source) == (
        '- Author. "Title." *arXiv:2506.05391*, '
        "2025. [arXiv](https://arxiv.org/abs/2506.05391).\n"
    )


def test_nested_year_numbered_items_remain_separate():
    source = (
        "- Releases:\n"
        "  2025. First release.\n"
        "  2026. Second release.\n"
    )

    assert unwrap_markdown_text(source) == source


def test_preserves_indented_code_beneath_list_item():
    source = (
        "- Run this\n"
        "  command:\n"
        '      print("hello")\n'
        "- Continue.\n"
    )

    assert unwrap_markdown_text(source) == (
        "- Run this command:\n"
        '      print("hello")\n'
        "- Continue.\n"
    )


def test_preserves_fenced_code_beneath_list_item():
    source = (
        "- Run this\n"
        "  example:\n"
        "  ```python\n"
        '  print("hello")\n'
        "  ```\n"
    )

    assert unwrap_markdown_text(source) == (
        "- Run this example:\n"
        "  ```python\n"
        '  print("hello")\n'
        "  ```\n"
    )


def test_preserves_explicit_line_break_in_list_item():
    source = "- First line  \n  second line\n"

    assert unwrap_markdown_text(source) == source


def test_preserves_backslash_line_break_in_list_item():
    source = "- First line\\\n  second line\n"

    assert unwrap_markdown_text(source) == source


def test_preserves_explicit_line_break_in_plain_paragraph():
    source = "First line\nsecond line  \nthird line\n"

    assert unwrap_markdown_text(source) == source


def test_preserves_latex_in_list_item_when_requested():
    source = "- Formula $x + y$ is\n  intentionally wrapped.\n"

    assert unwrap_markdown_text(source, preserve_latex=True) == source


def test_unwraps_latex_in_list_item_when_not_preserved():
    source = "- Formula $x + y$ is\n  intentionally wrapped.\n"

    assert unwrap_markdown_text(source, preserve_latex=False) == (
        "- Formula $x + y$ is intentionally wrapped.\n"
    )


def test_unwrapping_is_idempotent():
    source = "- First line\n  continued.\n- Next item.\n"
    once = unwrap_markdown_text(source)

    assert unwrap_markdown_text(once) == once
