"""Tests for statement and minutes parsing: body text only, sections, no page chrome."""

import re
from collections.abc import Callable
from datetime import date
from itertools import groupby

import pytest

from app.ingest.parse import (
    ATTENDANCE,
    FOOTNOTES,
    PREAMBLE,
    ParsedPage,
    normalize_text,
    parse_minutes,
    parse_statement,
)

FedPage = Callable[[str], bytes]
# Page furniture as it appears on the pages, as regexes ("PDFP" and "Share prices" are real text).
CHROME = [
    r"For (immediate )?release",
    r"For media inquiries",
    r"Implementation Note issued",
    r"Last Update",
    r"Return to text",
    r"Back to Top",
    r"email protected",
    r"(?m)^Share$",
    r"\bPDF\b",
    r"Federal Reserve issues FOMC statement",
    r"_{4,}",
]


def texts(page: ParsedPage) -> list[str]:
    return [paragraph.text for paragraph in page.paragraphs]


def assert_no_chrome(page: ParsedPage) -> None:
    joined = "\n".join(texts(page))
    for chrome in CHROME:
        assert not re.search(chrome, joined), chrome
    assert all(text and text == text.strip() and "  " not in text for text in texts(page))


def outline(page: ParsedPage) -> list[str | None]:
    return [section for section, _ in groupby(p.section for p in page.paragraphs)]


def test_normalize_text() -> None:
    nb_hyphen, nbsp, left_q, right_q, apostrophe = map(chr, (0x2011, 0xA0, 0x201C, 0x201D, 0x2019))
    em_dash, zero_width, quarter, ltr_mark = map(chr, (0x2014, 0x200B, 0xBC, 0x200E))
    raw = (
        f"4{nb_hyphen}1/4 percent,{nbsp}{left_q}ample{right_q} reserves {em_dash} the "
        f"Committee{apostrophe}s view{zero_width} {quarter}{ltr_mark}"
    )

    expected = f'4-1/4 percent, "ample" reserves {em_dash} the Committee\'s view 1/4'
    assert normalize_text(raw) == expected
    assert normalize_text("  many\n\n  spaces\t") == "many spaces"
    # a fraction sign after a digit keeps the Fed's ASCII form, not NFKC's '51/4'
    assert normalize_text("5\u00bc to 5\u00bd percent") == "5-1/4 to 5-1/2 percent"


def test_statement_2015(fed_page: FedPage) -> None:
    page = parse_statement(fed_page("monetary20150128a.htm"))

    assert page.page_date == date(2015, 1, 28)
    assert page.headline == "Federal Reserve issues FOMC statement"
    assert len(page.paragraphs) == 6
    assert texts(page)[0].startswith("Information received since the Federal Open Market")
    assert texts(page)[-1].startswith("Voting for the FOMC monetary policy action were: Janet L.")
    assert all(paragraph.section is None for paragraph in page.paragraphs)
    assert page.notes == []
    assert_no_chrome(page)


def test_statement_2025_keeps_dissent_and_drops_contact_and_links(fed_page: FedPage) -> None:
    page = parse_statement(fed_page("monetary20250917a.htm"))

    assert page.page_date == date(2025, 9, 17)
    assert len(page.paragraphs) == 5
    assert texts(page)[0].startswith("Recent indicators suggest that growth of economic activity")
    assert "4 to 4-1/4 percent" in texts(page)[2]  # non-breaking hyphen normalized
    assert texts(page)[-1].startswith("Voting for the monetary policy action were Jerome H.")
    assert texts(page)[-1].endswith("1/2 percentage point at this meeting.")
    assert_no_chrome(page)


def test_statement_2020_notation_vote_drops_board_announcements(fed_page: FedPage) -> None:
    page = parse_statement(fed_page("monetary20200323a.htm"))

    assert page.page_date == date(2020, 3, 23)
    assert len(page.paragraphs) == 3
    assert texts(page)[-1].startswith("Voting (by notation) for the monetary policy action")
    assert not any("In a related set of actions" in text for text in texts(page))
    assert not any("directs the Desk" in text for text in texts(page))
    assert page.notes == [
        "dropped 2 paragraph(s) after the vote, starting 'In a related set of actions, "
        "the Federal Reserve announced additional '"
    ]
    assert_no_chrome(page)


STATEMENT_WITH_DIRECTIVE = """
<div id="article"><div class="col-sm-8">
<p>The Federal Reserve will purchase Treasury bills at least into the second quarter of next
year.</p>
<p>In connection with these plans, the Federal Open Market Committee voted unanimously to authorize
and direct the Federal Reserve Bank of New York to execute transactions in accordance with the
following domestic policy directive:</p>
<p>"Effective October 15, 2019, the Federal Open Market Committee directs the Desk to purchase
Treasury bills."</p>
<p>More information about these plans may be found on the Federal Reserve Bank of New York's
website.</p>
</div></div>
"""


def test_statement_ends_at_the_quoted_directive_to_the_desk() -> None:
    page = parse_statement(STATEMENT_WITH_DIRECTIVE)

    assert texts(page) == [
        "The Federal Reserve will purchase Treasury bills at least into the second quarter of "
        "next year.",
        "In connection with these plans, the Federal Open Market Committee voted unanimously to "
        "authorize and direct the Federal Reserve Bank of New York to execute transactions in "
        "accordance with the following domestic policy directive:",
    ]


def test_statement_2026_vote_tally_format(fed_page: FedPage) -> None:
    page = parse_statement(fed_page("monetary20260729a.htm"))

    assert len(page.paragraphs) == 5
    assert texts(page)[0] == (
        "The Federal Open Market Committee approved the following statement for release by a "
        "9 - 3 vote:"
    )
    assert texts(page)[-1].startswith("Voting against the monetary policy action were Beth M.")
    assert page.notes == []
    assert_no_chrome(page)


def test_minutes_2016_sections_and_roster(fed_page: FedPage) -> None:
    page = parse_minutes(fed_page("fomcminutes20160127.htm"))

    assert page.headline == "Minutes of the Federal Open Market Committee"
    assert outline(page) == [
        PREAMBLE,
        ATTENDANCE,
        "Annual Organizational Matters",
        "Developments in Financial Markets, Open Market Operations, and Policy Normalization",
        "Staff Review of the Economic Situation",
        "Staff Review of the Financial Situation",
        "Staff Economic Outlook",
        "Participants' Views on Current Conditions and the Economic Outlook",
        "Committee Policy Action",
        "Potential Enhancements to the Summary of Economic Projections",
        "Notation Vote",
        FOOTNOTES,
    ]
    preamble = [p.text for p in page.paragraphs if p.section == PREAMBLE]
    assert len(preamble) == 1 and preamble[0].startswith("A joint meeting of the Federal Open")
    roster = [p.text for p in page.paragraphs if p.section == ATTENDANCE]
    assert roster[0].startswith("Janet L. Yellen, Chair; William C. Dudley, Vice Chairman; Lael")
    # the footnote marker after "Ann McKeehan," is gone
    assert any("Andrew Figura, Ann McKeehan, David Reifschneider" in text for text in roster)
    assert all(len(text.split()) < 60 for text in roster)
    assert page.notes == ["dropped signature block: 'Brian F. Madigan / Secretary'"]
    assert page.warnings == []
    assert_no_chrome(page)


def test_minutes_2016_tables_and_footnotes(fed_page: FedPage) -> None:
    page = parse_minutes(fed_page("fomcminutes20160127.htm"))

    assert any(
        text.startswith("Janet L. Yellen, Chairman; William C. Dudley, Vice Chairman; Brian F.")
        for text in texts(page)
    )  # the officers table, one "name, title" per row
    footnotes = [p.text for p in page.paragraphs if p.section == FOOTNOTES]
    assert len(footnotes) == 5
    assert footnotes[0].startswith("The Federal Open Market Committee is referenced as the")
    assert not any(text[0].isdigit() for text in footnotes)
    assert sum(len(text.split()) for text in texts(page)) > 10_000


def test_minutes_2025_attendance_at_end(fed_page: FedPage) -> None:
    page = parse_minutes(fed_page("fomcminutes20250917.htm"))

    sections = outline(page)
    assert sections[:2] == [
        PREAMBLE,
        "Developments in Financial Markets and Open Market Operations",
    ]
    assert sections[-3:] == ["Notation Vote", ATTENDANCE, FOOTNOTES]
    assert "Committee Policy Actions" in sections
    assert "Voting for this action:" not in sections
    assert 100 < len(page.paragraphs) < 200
    roster = [p.text for p in page.paragraphs if p.section == ATTENDANCE]
    assert roster[0].startswith("Jerome H. Powell, Chair; John C. Williams, Vice Chair;")
    assert roster[-1] == "Andrei Zlate, Group Manager, Division of Monetary Affairs, Board"
    assert len([p for p in page.paragraphs if p.section == FOOTNOTES]) == 4
    assert page.notes == ["dropped signature block: 'Joshua Gallin / Secretary'"]
    assert_no_chrome(page)


def test_minutes_2025_policy_action_keeps_directive_vote_and_statement(fed_page: FedPage) -> None:
    page = parse_minutes(fed_page("fomcminutes20250917.htm"))
    policy = [p.text for p in page.paragraphs if p.section == "Committee Policy Actions"]

    assert (
        '"Effective September 18, 2025, the Federal Open Market Committee directs the Desk to:'
        in (policy)
    )
    assert any(text.startswith("Undertake open market operations as necessary") for text in policy)
    assert any(text.startswith('"Recent indicators suggest that growth') for text in policy)
    assert "Voting against this action: Stephen I. Miran." in policy
    voting_for = next(text for text in policy if text.startswith("Voting for this action:"))
    assert voting_for.endswith("and Christopher J. Waller.")


SYNTHETIC_OLD_LAYOUT = """
<div id="article"><h3>Minutes of the Federal Open Market Committee</h3>
<p><strong>June 9</strong><strong>&ndash;10, 2020</strong></p>
<p>A joint meeting was held by videoconference.<a href="#fn1"><sup>1</sup></a><a name="f1"></a></p>
<p>PRESENT:<br />Jerome H. Powell, Chair<br />John C. Williams, Vice Chair</p>
<p>Daniel Cooper, Senior Economist, Federal Reserve Bank of Boston</p>
<p>The Chair opened the meeting with remarks. He noted recent events. All agreed.</p>
<blockquote>A. Arrangements with the following foreign banks:
<table><tr><td>Bank of Canada</td><td>2,000</td></tr><tr><td>Bank of Mexico</td><td>3,000</td></tr>
</table></blockquote>
<blockquote>Bank of Canada<br />Bank of England<br />Bank of Japan</blockquote>
<p><strong>Committee Policy Action</strong><br />Members agreed.</p>
<p><strong>Voting for this action</strong>: Jerome H. Powell.</p>
<center><p>_____________</p>
<strong>James A. Clouse</strong><br /><strong>Secretary</strong></center>
<hr /><p><a name="fn1a"></a>1. Attended by videoconference. <a href="#f1">Return to text</a></p>
</div>
"""


def test_minutes_older_layout_edge_cases() -> None:
    page = parse_minutes(SYNTHETIC_OLD_LAYOUT)

    assert [(p.section, p.text) for p in page.paragraphs] == [
        (PREAMBLE, "A joint meeting was held by videoconference."),
        (ATTENDANCE, "Jerome H. Powell, Chair; John C. Williams, Vice Chair"),
        (
            ATTENDANCE,
            "Daniel Cooper, Senior Economist, Federal Reserve Bank of Boston",
        ),
        (
            PREAMBLE,
            "The Chair opened the meeting with remarks. He noted recent events. All agreed.",
        ),
        (PREAMBLE, "A. Arrangements with the following foreign banks:"),
        (PREAMBLE, "Bank of Canada, 2,000; Bank of Mexico, 3,000"),
        (PREAMBLE, "Bank of Canada; Bank of England; Bank of Japan"),
        ("Committee Policy Action", "Members agreed."),
        ("Committee Policy Action", "Voting for this action: Jerome H. Powell."),
        (FOOTNOTES, "Attended by videoconference."),
    ]
    assert page.notes == ["dropped signature block: 'James A. Clouse / Secretary'"]


def test_page_updated_comes_from_the_last_update_stamp() -> None:
    article = '<div id="article"><div class="col-sm-8"><p>Rates were unchanged.</p></div></div>'
    stamp = '<div class="lastUpdate" id="lastUpdate">Last Update:\n    October 09, 2026\n</div>'

    assert parse_statement(article + stamp).page_updated == date(2026, 10, 9)
    assert parse_minutes(article + stamp).page_updated == date(2026, 10, 9)
    assert parse_statement(article).page_updated is None


def test_pages_without_an_article_are_rejected() -> None:
    with pytest.raises(ValueError, match="#article"):
        parse_statement("<html><body><p>Not found</p></body></html>")
    with pytest.raises(ValueError, match="#article"):
        parse_minutes("<html><body></body></html>")


def test_minutes_html_headings_and_text_after_signature_are_flagged() -> None:
    html = """<div id="article">
    <p>PRESENT:<br />Jerome H. Powell, Chair</p>
    <h5>Staff Economic Outlook</h5><p>The staff projection was little changed.</p>
    <p>_______________________</p>
    <p>Joshua Gallin<br />Secretary</p>
    <p>Participants agreed that the outlook was uncertain and that risks were balanced.</p>
    </div>"""

    page = parse_minutes(html)

    assert [(p.section, p.text) for p in page.paragraphs] == [
        (ATTENDANCE, "Jerome H. Powell, Chair"),
        ("Staff Economic Outlook", "The staff projection was little changed."),
    ]
    assert len(page.warnings) == 1
    assert page.warnings[0].startswith("text after the signature line was dropped")
