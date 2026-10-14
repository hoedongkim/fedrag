"""Turn FOMC statement and minutes HTML into clean paragraphs, dropping page chrome."""

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date

from bs4 import BeautifulSoup, NavigableString, Tag
from bs4.element import PreformattedString

from app.ingest.calendar import parse_release_date
from app.ingest.records import Paragraph

PREAMBLE = "Preamble"
ATTENDANCE = "Attendance"
FOOTNOTES = "Footnotes"

# Typographic variants become ASCII so text matches however it was typed: hyphens, quotes and
# the fraction slash NFKC makes of any fraction sign not in FRACTIONS. Invisible characters (soft
# hyphen, zero-width and bidi marks, byte-order mark) are removed. Em dashes are kept.
PUNCTUATION = str.maketrans(
    {
        **dict.fromkeys("\u2010\u2011\u2012\u2013\u2212", "-"),
        **dict.fromkeys("\u2018\u2019\u201a\u201b\u2032", "'"),
        **dict.fromkeys("\u201c\u201d\u201e\u2033", '"'),
        "\u2044": "/",
        **dict.fromkeys("\u00ad\u200b\u200c\u200d\u200e\u200f"),
        **dict.fromkeys("\u2060\u202a\u202b\u202c\u202d\u202e\ufeff"),
    }
)
# Vulgar fractions in the Fed's ASCII style: '4¼' -> '4-1/4', a lone '¼' -> '1/4'. NFKC alone
# would turn '4¼' into '41/4'.
FRACTIONS = {"¼": "1/4", "½": "1/2", "¾": "3/4", "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8"}
FRACTION_RE = re.compile(r"(\d?)([¼½¾⅛-⅞])")
# A statement body ends at the first of these: contact line, related releases, the quoted
# directive to the Desk, page furniture.
STATEMENT_END_RE = re.compile(
    r"^(For media inquiries|Implementation Note|Attachments?\b|Related Content|Last Update"
    r"|More information (about|regarding|on|can be found)|\"Effective )",
    re.IGNORECASE,
)
VOTING_FOR_RE = re.compile(r"^Voting (\(by notation\) )?for\b")
VOTING_MORE_RE = re.compile(r"^(Voting against|Absent and not voting)\b")
DATE_LINE_RE = re.compile(r"^[A-Z][a-z]+\.? \d{1,2}(-([A-Z][a-z]+\.? )?\d{1,2})?, \d{4}$")
ATTENDANCE_START_RE = re.compile(r"^PRESENT\s*:?\s*(.*)$")
SIGNATURE_RE = re.compile(r"^_{5,}")
FOOTNOTE_NUMBER_RE = re.compile(r"^\d{1,2}\.?\s+")
FOOTNOTE_ANCHOR_RE = re.compile(r"^fn\d+[a-z]?$")
MAX_HEADING_WORDS = 16
MAX_LIST_LINE_WORDS = 10
MAX_SIGNATURE_WORDS = 8  # 'Joshua Gallin / Secretary'; more means content was swallowed
SENTENCE_BREAK_RE = re.compile(r"[a-z]{3,}[.?!]\"? [A-Z]")
SIGNATURE_NOTE = "dropped signature block"


def normalize_text(text: str) -> str:
    """Fed-style fractions, NFKC, ASCII hyphens/quotes, no invisible characters, single spaces."""
    text = FRACTION_RE.sub(lambda m: (m[1] + "-" if m[1] else "") + FRACTIONS[m[2]], text)
    text = unicodedata.normalize("NFKC", text).translate(PUNCTUATION)
    return " ".join(text.split())


@dataclass
class ParsedPage:
    """What a parser extracted from one page, plus notes on anything it dropped or doubted."""

    paragraphs: list[Paragraph]
    page_date: date | None = None
    page_updated: date | None = None
    headline: str | None = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _article(html: bytes | str) -> tuple[Tag, date | None]:
    """The page's #article element without scripts or footnote markers, and its update stamp."""
    soup = BeautifulSoup(html, "lxml")
    article = soup.find(id="article")
    if not isinstance(article, Tag):
        raise ValueError("Page has no #article element")
    for node in article.find_all(["script", "style", "noscript"]):
        node.decompose()
    _drop_footnote_markers(article)
    return article, _last_update(soup)


def _last_update(soup: BeautifulSoup) -> date | None:
    """The page's 'Last Update: October 09, 2026' stamp, printed below #article."""
    stamp = soup.find(id="lastUpdate")
    text = (
        normalize_text(stamp.get_text(" ")).removeprefix("Last Update:").strip()
        if isinstance(stamp, Tag)
        else ""
    )
    try:
        return parse_release_date(text) if text else None
    except (KeyError, ValueError):  # KeyError: a word that is not a month name
        return None


def _drop_footnote_markers(root: Tag) -> None:
    """Remove in-text footnote references, bare <sup>1</sup> markers and "Return to text" links."""
    for anchor in root.find_all("a", href=re.compile(r"^#fn\d+[a-z]?$")):
        anchor.decompose()
    for sup in root.find_all("sup"):
        if re.fullmatch(r"[\d,*\s]+", sup.get_text()):
            sup.decompose()
    for anchor in root.find_all("a", href=re.compile(r"^#f\d+[a-z]?$")):
        if "return to text" in anchor.get_text(" ", strip=True).lower():
            anchor.decompose()


def _lines(node: Tag) -> list[tuple[str, bool]]:
    """Split a block on <br> into normalized lines; the flag is True when a line is all bold."""
    lines: list[tuple[str, bool]] = []
    parts: list[tuple[str, bool]] = []

    def flush() -> None:
        text = normalize_text("".join(part for part, _ in parts))
        if text:
            lines.append((text, all(bold for part, bold in parts if part.strip())))
        parts.clear()

    def walk(parent: Tag, bold: bool) -> None:
        for child in parent.children:
            if isinstance(child, Tag):
                if child.name == "br":
                    flush()
                    continue
                child_bold = bold or child.name in ("strong", "b")
                block = child.name in BLOCK_TAGS
                if block:
                    parts.append((" ", child_bold))
                walk(child, child_bold)
                if block:
                    parts.append((" ", child_bold))
            elif isinstance(child, NavigableString) and not isinstance(child, PreformattedString):
                parts.append((str(child), bold))

    walk(node, False)
    flush()
    return lines


BLOCK_TAGS = frozenset({"p", "div", "li", "ul", "ol", "td", "th", "tr", "blockquote", "center"})


def _table_text(table: Tag) -> str:
    """One line per table row ('Janet L. Yellen, Chair'), rows joined with semicolons."""
    rows = []
    for row in table.find_all("tr"):
        cells = [normalize_text(cell.get_text(" ")) for cell in row.find_all(["td", "th"])]
        if text := ", ".join(cell for cell in cells if cell):
            rows.append(text)
    return "; ".join(rows)


def _blocks(container: Tag) -> Iterator[Tag]:
    """Paragraph-level blocks in document order: paragraphs, list items, tables, quoted text."""
    _wrap_loose_inline(container)
    for child in container.children:
        if not isinstance(child, Tag):
            continue
        classes = child.get("class") or []
        if child.name in ("p", "h4", "h5", "h6", "table") or "footnotes" in classes:
            yield child
        elif child.name in ("ul", "ol"):
            yield from child.find_all("li", recursive=False)
        elif child.name in CONTAINER_TAGS and "panel" not in classes:
            if child.find(["p", "li", "table", "div"]):
                yield from _blocks(child)
            else:
                yield child


CONTAINER_TAGS = frozenset({"blockquote", "div", "center", "section"})
BLOCK_LEVEL = BLOCK_TAGS | CONTAINER_TAGS | {"table", "h1", "h2", "h3", "h4", "h5", "h6", "hr"}


def _wrap_loose_inline(container: Tag) -> None:
    """Wrap text that sits beside block children (e.g. a caption before a table) in a <p>."""
    run: list[Tag | NavigableString] = []

    def flush() -> None:
        if any(
            normalize_text(node.get_text() if isinstance(node, Tag) else str(node))
            for node in run
            if not isinstance(node, PreformattedString)
        ):
            wrapper = BeautifulSoup("", "lxml").new_tag("p")
            run[0].insert_before(wrapper)
            for node in run:
                wrapper.append(node.extract())
        run.clear()

    if not any(isinstance(c, Tag) and c.name in BLOCK_LEVEL for c in container.children):
        return
    for child in list(container.children):
        if isinstance(child, Tag) and child.name in BLOCK_LEVEL:
            flush()
        else:
            run.append(child)
    flush()


def _is_heading(text: str) -> bool:
    return (
        len(text.split()) <= MAX_HEADING_WORDS
        and text[:1].isupper()
        and not text.endswith((":", ".", ";", ","))
        and not SIGNATURE_RE.match(text)
    )


# --- statements ------------------------------------------------------------------------------


def parse_statement(html: bytes | str) -> ParsedPage:
    """Statement body from a press release: the policy paragraphs and the vote, nothing else."""
    article, page_updated = _article(html)
    page = ParsedPage(paragraphs=[], page_updated=page_updated)
    if (time_tag := article.find(class_="article__time")) is not None:
        page.page_date = parse_release_date(normalize_text(time_tag.get_text(" ")))
    if (title_tag := article.find("h3")) is not None:
        page.headline = normalize_text(title_tag.get_text(" "))

    body = next(
        (
            div
            for div in article.find_all("div", recursive=False)
            if "col-sm-8" in (div.get("class") or []) and "heading" not in div["class"]
        ),
        None,
    )
    if body is None:
        raise ValueError("Statement page has no body column")

    texts: list[str] = []
    for block in _blocks(body):
        text = _table_text(block) if block.name == "table" else _joined(block)
        if not text:
            continue
        if STATEMENT_END_RE.match(text):
            break
        texts.append(text)

    voting = next((i for i, text in enumerate(texts) if VOTING_FOR_RE.match(text)), None)
    if voting is not None:
        end = voting + 1
        while end < len(texts) and VOTING_MORE_RE.match(texts[end]):
            end += 1
        if end < len(texts):
            page.notes.append(
                f"dropped {len(texts) - end} paragraph(s) after the vote, starting "
                f"{texts[end][:70]!r}"
            )
        texts = texts[:end]
    elif not any(VOTING_MORE_RE.match(text) or re.search(r"\bvote\b", text) for text in texts):
        page.notes.append("no voting paragraph")
    page.paragraphs = [Paragraph(section=None, text=text) for text in texts]
    return page


def _joined(block: Tag) -> str:
    return _join_lines([line for line, _ in _lines(block)])


def _join_lines(lines: list[str], list_like: bool = False) -> str:
    """Join <br>-separated lines: with "; " for rosters and short list lines, else with spaces."""
    short = len(lines) > 1 and all(len(line.split()) <= MAX_LIST_LINE_WORDS for line in lines)
    return ("; " if list_like or short else " ").join(lines)


# --- minutes ---------------------------------------------------------------------------------


def parse_minutes(html: bytes | str) -> ParsedPage:
    """Minutes as paragraphs labelled with their section heading.

    The opening date line is dropped (it is metadata). Text before the first heading is the
    "Preamble"; the participant roster (after "PRESENT:" or under an "Attendance" heading) is the
    "Attendance" section and ends at the first heading or prose paragraph; footnotes form a
    "Footnotes" section; the Secretary's signature block is dropped.
    """
    article, page_updated = _article(html)
    page = ParsedPage(paragraphs=[], page_updated=page_updated)
    if (title_tag := article.find("h3")) is not None:
        page.headline = normalize_text(title_tag.get_text(" "))

    section = resume = PREAMBLE  # `resume`: where prose after the roster belongs
    after_signature: list[str] = []
    in_signature = False

    def add(section_name: str, text: str) -> None:
        if text:
            page.paragraphs.append(Paragraph(section=section_name, text=text))

    for block in _blocks(article):
        if "footnotes" in (block.get("class") or []) or _is_footnote(block):
            notes = block.find_all("p") if block.name == "div" else [block]
            for note in notes or [block]:
                add(FOOTNOTES, FOOTNOTE_NUMBER_RE.sub("", _joined(note), count=1))
            continue
        if block.name == "table":
            add(section, _table_text(block))
            continue
        lines = _lines(block)
        if not lines:
            continue
        first, bold = lines[0]
        bold = bold or block.name in ("h4", "h5", "h6")
        if SIGNATURE_RE.match(first) or in_signature:
            in_signature = True
            after_signature.extend(line for line, _ in lines if not SIGNATURE_RE.match(line))
            continue
        if bold and section == PREAMBLE and DATE_LINE_RE.match(first):
            lines = lines[1:]
        elif section == PREAMBLE and (present := ATTENDANCE_START_RE.match(first)):
            resume, section = section, ATTENDANCE
            lines = [(present[1], False), *lines[1:]] if present[1] else lines[1:]
        elif bold and _is_heading(first):
            lines = lines[1:]
            if first.lower() == "attendance":
                resume, section = section, ATTENDANCE
            else:
                section = first
        texts = [line for line, _ in lines]
        if section == ATTENDANCE and _is_prose(_join_lines(texts)):
            section = resume
        add(section, _join_lines(texts, list_like=section == ATTENDANCE))

    if after_signature:
        dropped = " / ".join(after_signature)
        page.notes.append(f"{SIGNATURE_NOTE}: {dropped[:120]!r}")
        if len(dropped.split()) > MAX_SIGNATURE_WORDS:
            page.warnings.append(f"text after the signature line was dropped: {dropped[:120]!r}")
    if not any(p.section == ATTENDANCE for p in page.paragraphs):
        page.warnings.append("no attendance roster found")
    return page


def _is_prose(text: str) -> bool:
    """True for running text (a sentence boundary, or a long sentence), not a roster entry."""
    words = len(text.split())
    return bool(SENTENCE_BREAK_RE.search(text)) or (text.endswith(".") and words >= 25)


def _is_footnote(block: Tag) -> bool:
    """A footnote paragraph starts with its anchor: <p><a name="fn1">1. </a>Text ...</p>."""
    anchor = block.find("a", attrs={"name": FOOTNOTE_ANCHOR_RE}) or block.find(
        "a", attrs={"id": FOOTNOTE_ANCHOR_RE}
    )
    if anchor is None:
        return False
    for item in block.descendants:
        if item is anchor:
            return True
        if isinstance(item, NavigableString) and item.strip():
            return False
    return False
