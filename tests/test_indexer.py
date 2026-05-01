"""Tests for the indexer's heading-aware HTML chunking."""

from src.indexer.run_indexer import (
    _format_heading_prefix,
    _is_stale_section,
    _current_academic_year_start,
    extract_html_chunks,
    chunk_text,
)

# ── _format_heading_prefix ────────────────────────────────────


def test_heading_prefix_single_level():
    assert _format_heading_prefix({1: "Fees"}) == "Fees"


def test_heading_prefix_multi_level():
    stack = {1: "Admissions", 2: "Main School Fees", 3: "2025/2026"}
    assert _format_heading_prefix(stack) == "Admissions > Main School Fees > 2025/2026"


def test_heading_prefix_skipped_levels():
    """h1 then h3 (no h2) should still concatenate in level order."""
    stack = {1: "Fees", 3: "Year 2025/26"}
    assert _format_heading_prefix(stack) == "Fees > Year 2025/26"


def test_heading_prefix_empty_stack():
    assert _format_heading_prefix({}) == ""


# ── extract_html_chunks — basic heading context ───────────────


def test_chunks_prepend_h2_heading():
    html = b"""
    <html><body>
      <h2>About Us</h2>
      <p>Warwick Prep is an independent school.</p>
    </body></html>
    """
    chunks = extract_html_chunks(html)
    assert len(chunks) == 1
    assert chunks[0].startswith("[About Us]")
    assert "Warwick Prep is an independent school" in chunks[0]


def test_chunks_prepend_breadcrumb_for_nested_headings():
    html = b"""
    <html><body>
      <h2>School Fees</h2>
      <h3>2025/2026</h3>
      <p>Reception: &pound;4,509 per term.</p>
    </body></html>
    """
    chunks = extract_html_chunks(html)
    assert any("School Fees > 2025/2026" in c for c in chunks)
    assert any("Reception" in c and "4,509" in c for c in chunks)


def test_chunks_separate_sections_under_different_headings():
    """Stale-year sections are suppressed; only the current-year section survives."""
    html = b"""
    <html><body>
      <h2>Fees 2024/2025</h2>
      <p>Reception: &pound;3,950 per term.</p>
      <h2>Fees 2025/2026</h2>
      <p>Reception: &pound;4,509 per term.</p>
    </body></html>
    """
    chunks = extract_html_chunks(html)
    old_chunks = [c for c in chunks if "3,950" in c]
    new_chunks = [c for c in chunks if "4,509" in c]

    assert not old_chunks, "2024/25 fee chunk must be suppressed by stale-section filter"
    assert new_chunks, "2025/26 fee chunk should exist"
    assert all("Fees 2025/2026" in c for c in new_chunks)


def test_chunks_heading_reset_clears_deeper_levels():
    """A same-level h2 should replace — not append to — the previous h2 context."""
    html = b"""
    <html><body>
      <h2>Section A</h2>
      <h3>Subsection A1</h3>
      <p>Content A1.</p>
      <h2>Section B</h2>
      <p>Content B.</p>
    </body></html>
    """
    chunks = extract_html_chunks(html)
    # Section B content must NOT include "Section A" or "Subsection A1" in its prefix
    b_chunks = [c for c in chunks if "Content B" in c]
    assert b_chunks
    assert all("Section A" not in c for c in b_chunks)
    assert all("Subsection A1" not in c for c in b_chunks)
    assert all("Section B" in c for c in b_chunks)


def test_chunks_no_headings_plain_text():
    """Pages with no headings should still return chunks (without any prefix)."""
    html = b"""
    <html><body>
      <p>Welcome to Warwick Prep School. We are an outstanding prep school.</p>
    </body></html>
    """
    chunks = extract_html_chunks(html)
    assert len(chunks) == 1
    assert (
        chunks[0]
        == "Welcome to Warwick Prep School. We are an outstanding prep school."
    )


# ── extract_html_chunks — tables ─────────────────────────────


def test_chunks_table_emitted_as_pipe_rows():
    """Table cells should be pipe-separated so fee rows stay coherent."""
    html = b"""
    <html><body>
      <h2>2025/2026 Fees</h2>
      <table>
        <tr><th>Year Group</th><th>Net Fee</th><th>Total</th></tr>
        <tr><td>Reception</td><td>&pound;4,509</td><td>&pound;5,411</td></tr>
        <tr><td>Years 3 &amp; 4</td><td>&pound;5,205</td><td>&pound;6,246</td></tr>
      </table>
    </body></html>
    """
    chunks = extract_html_chunks(html)
    assert any(
        "Reception | \u00a34,509 | \u00a35,411" in c for c in chunks
    ), "Reception row should appear as pipe-separated within a chunk"
    assert any("Years 3 & 4 | \u00a35,205 | \u00a36,246" in c for c in chunks)


def test_chunks_table_inherits_heading_prefix():
    """Table chunks must carry the nearest heading breadcrumb."""
    html = b"""
    <html><body>
      <h2>Main School Fees</h2>
      <h3>Fees for the Academic Year 2025/2026</h3>
      <table>
        <tr><td>Years 5 &amp; 6</td><td>&pound;5,526</td></tr>
      </table>
    </body></html>
    """
    chunks = extract_html_chunks(html)
    assert any(
        "Main School Fees > Fees for the Academic Year 2025/2026" in c and "5,526" in c
        for c in chunks
    ), "Fee table chunk must carry the full heading breadcrumb"


# ── extract_html_chunks — real-world fees page scenario ──────


def test_chunks_non_year_sections_both_kept():
    """Sections without year patterns in their headings are never filtered."""
    html = b"""
    <html><body>
      <h2>Prep Department</h2>
      <p>Years 3 to 6.</p>
      <h2>Pre-Prep Department</h2>
      <p>Reception to Year 2.</p>
    </body></html>
    """
    chunks = extract_html_chunks(html)
    assert any("Years 3 to 6" in c for c in chunks)
    assert any("Reception to Year 2" in c for c in chunks)


def test_chunks_fee_page_two_years_correctly_labelled():
    """With stale-section suppression, only the current year's fees appear in chunks.
    The 2024/25 section must be silently dropped; the 2025/26 section must survive.
    """
    html = b"""
    <html><body>
      <h1>Fees</h1>
      <h2>Fees for the Academic Year 2025/2026</h2>
      <table>
        <tr><td>Reception, Years 1 &amp; 2</td><td>Net Fee &pound;4,509</td><td>Total &pound;5,411</td></tr>
        <tr><td>Years 3 &amp; 4</td><td>Net Fee &pound;5,205</td><td>Total &pound;6,246</td></tr>
      </table>
      <h2>Main School Fees (Per term)</h2>
      <h3>Fees for the Academic Year 2024/2025 From 1 January 2025:</h3>
      <table>
        <tr><td>Reception, Years 1 and 2</td><td>&pound;3,950</td></tr>
        <tr><td>Years 3 and 4</td><td>&pound;4,560</td></tr>
      </table>
    </body></html>
    """
    chunks = extract_html_chunks(html)

    new_fee_chunks = [c for c in chunks if "4,509" in c or "5,205" in c]
    old_fee_chunks = [c for c in chunks if "3,950" in c or "4,560" in c]

    assert new_fee_chunks, "2025/26 fee data must appear in chunks"
    assert not old_fee_chunks, "2024/25 fee data must be suppressed by stale-section filter"


# ── extract_html_chunks — nav/footer stripped ────────────────


def test_chunks_strips_nav_and_footer():
    html = b"""
    <html><body>
      <nav><a href="/">Home</a><a href="/admissions">Admissions</a></nav>
      <h2>Contact Us</h2>
      <p>Call 01926 491545.</p>
      <footer>Copyright 2026 Warwick Prep School</footer>
    </body></html>
    """
    chunks = extract_html_chunks(html)
    combined = " ".join(chunks)
    assert "Home" not in combined
    assert "Copyright" not in combined
    assert "01926 491545" in combined


def test_chunks_strips_script_and_style():
    html = b"""
    <html><head>
      <style>.foo { color: red; }</style>
      <script>alert('test');</script>
    </head><body>
      <h2>Admissions</h2>
      <p>Register today.</p>
    </body></html>
    """
    chunks = extract_html_chunks(html)
    combined = " ".join(chunks)
    assert "color: red" not in combined
    assert "alert" not in combined
    assert "Register today" in combined


# ── extract_html_chunks — chunking size ──────────────────────


def test_chunks_long_section_split_into_multiple_chunks():
    """A section with more than chunk_size words must produce multiple chunks."""
    words = " ".join(f"word{i}" for i in range(600))
    html = f"<html><body><h2>Long Section</h2><p>{words}</p></body></html>".encode()
    chunks = extract_html_chunks(html, chunk_size=512, overlap=64)
    assert len(chunks) > 1
    # Every chunk must carry the heading prefix
    assert all("[Long Section]" in c for c in chunks)


def test_chunks_overlap_shared_words_between_adjacent_chunks():
    """Adjacent chunks of a long section must share the overlap words."""
    words = [f"word{i}" for i in range(600)]
    html = f"<html><body><h2>S</h2><p>{' '.join(words)}</p></body></html>".encode()
    chunks = extract_html_chunks(html, chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    # Words from the end of chunk 0 must appear at the start of chunk 1 (after the prefix)
    chunk0_words = chunks[0].split()[-20:]
    chunk1_content = chunks[1].split("[S] ", 1)[-1]  # strip prefix
    chunk1_first_words = chunk1_content.split()[:20]
    overlap_found = any(w in chunk1_first_words for w in chunk0_words)
    assert overlap_found, "Adjacent chunks must share overlapping words"


# ── _is_stale_section ────────────────────────────────────────


def test_stale_section_detects_past_academic_year():
    """Headings naming a past academic year should be flagged stale."""
    assert _is_stale_section("Fees for the Academic Year 2024/2025")
    assert _is_stale_section("Fees 2024/25")


def test_stale_section_not_stale_for_current_year():
    """The current academic year should not be flagged as stale."""
    current = _current_academic_year_start()
    assert not _is_stale_section(
        f"Fees for the Academic Year {current}/{current + 1}"
    )


def test_stale_section_detects_last_year_phrase():
    assert _is_stale_section("Last Year's Fees")
    assert _is_stale_section("Previous Year Fees")


def test_stale_section_not_stale_for_no_year():
    assert not _is_stale_section("Main School Fees")
    assert not _is_stale_section("Contact Us")
    assert not _is_stale_section("")


def test_stale_section_not_stale_future_year():
    """A section for a future academic year must never be filtered."""
    current = _current_academic_year_start()
    assert not _is_stale_section(
        f"Fees for the Academic Year {current + 1}/{current + 2}"
    )


# ── chunk_text (kept for PDF path) ───────────────────────────


def test_chunk_text_basic_split():
    words = " ".join(str(i) for i in range(100))
    chunks = chunk_text(words, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_chunk_text_single_chunk_when_short():
    text = "Short text that fits in one chunk."
    chunks = chunk_text(text, chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_empty_returns_empty():
    assert chunk_text("", chunk_size=512, overlap=64) == []
