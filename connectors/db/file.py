import logging
import os
import re
import string
from io import BytesIO, StringIO
from typing import Optional

import html2text
import mdformat
import PyPDF2
import pytablereader as ptr
from bs4 import BeautifulSoup
from docling.document_converter import DocumentConverter, DocumentStream
from tabledata import TableData

log = logging.getLogger("tangerine.file")

# match example: [Text](something)
LINK_REGEX = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# match example: "http://something.com"
ABSOLUTE_URL_REGEX = re.compile(r"[a-z0-9]*:\/\/.*")


def validate_file_path(full_path: str) -> None:
    if not isinstance(full_path, str):
        raise TypeError(f"file path must be a string, not {type(full_path)}")
    elif not full_path.strip():
        raise ValueError("file path cannot be empty string")
    else:
        # https://stackoverflow.com/a/73659000
        invalid_chars = set('\\?%*:|"<>')

        for char in invalid_chars:
            if char in full_path:
                raise ValueError(f"file path cannot contain characters: {invalid_chars}")


def validate_source(source: str) -> None:
    source_regex = r"^[\w-]+$"
    if not source or not source.strip() or not re.match(source_regex, source):
        raise ValueError(f"source must match regex: {source_regex}")


def validate_file_type(full_path: str) -> None:
    if not any(
        [
            full_path.endswith(filetype)
            for filetype in [".txt", ".pdf", ".md", ".rst", ".html", ".adoc"]
        ]
    ):
        raise ValueError("unsupported file type")


def _remove_large_md_code_blocks(text):
    """
    Replaces markdown code blocks longer than 9 lines with redirection text

    This is to avoid large code blocks getting broken up across text chunks
    """
    lines = []
    code_lines = []
    in_code_block = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```") and not in_code_block:
            in_code_block = True
            code_lines = []
            code_lines.append(line)
        elif line.lstrip().startswith("```") and in_code_block:
            code_lines.append(line)
            in_code_block = False
            if len(code_lines) > 9:
                # remove this block because it is too long, but preserve indentation of the block
                whitespace = " " * (len(line) - len(line.lstrip()))
                code_lines = [
                    line,
                    f"{whitespace}<large code block, visit documentation to view>",
                    line,
                ]
            lines.extend(code_lines)
        elif in_code_block:
            code_lines.append(line)
        else:
            lines.append(line)

    return "\n".join(lines)


def _adoc_to_md(path, text):
    converter = DocumentConverter()
    textio = BytesIO(text.encode("utf-8"))
    textio.seek(0)
    ds = DocumentStream(name=path, stream=textio)
    result = converter.convert(ds)
    return result.document.export_to_markdown()


def _get_table_row_lines(table: TableData) -> list[str]:
    table_lines = []
    for _, rows in table.as_dict().items():
        for row in rows:
            line = "* " + " \t| ".join(
                [f"{header}: {row_content or 'null'}" for header, row_content in row.items()]
            )
            table_lines.append(line)

    return table_lines


def _convert_md_tables(text: str) -> str:
    """
    Converts tables into plain with with each row having "header: value" statements

    Intended to preserve the context of the table headers across text chunks
    """
    # parse tables found in this text using pytablereader
    table_loader = ptr.MarkdownTableTextLoader(text)
    tables = table_loader.load()
    table_for_regex = dict()
    for table in tables:
        # create a regex pattern to match: '| header1   | header2   | (and so on)... |'
        headers = [re.escape(header) for header in table.headers]
        re_str = r"\| " + r"[\t ]+\| ".join(headers) + r"[\t ]+\|"
        table_for_regex[re_str] = table

    line_num = 0
    lines = text.split("\n")
    new_lines = []

    while line_num < len(lines):
        line = lines[line_num]
        for re_str in table_for_regex:
            line_without_styling = re.sub(r"[*_~]", "", line)
            if re.findall(re_str, line_without_styling):
                # we found the start of a table
                table = table_for_regex[re_str]
                row_lines = _get_table_row_lines(table)
                new_lines.append(
                    "<the table below was condensed using 'header: value' format for rows>"
                )
                new_lines.extend(row_lines)
                # skip over table rows and 2 header lines when continuing to process the text
                line_num += len(row_lines) + 2
                break
        else:
            new_lines.append(line)
            line_num += 1

    return "\n".join(new_lines)


def _convert_relative_links(md: str, url: str) -> str:
    url_prefix = url.rstrip(os.path.basename(url))  # remove filename at end

    if not url_prefix.endswith("/"):
        url_prefix = url_prefix + "/"

    md_lines = md.splitlines()

    for idx, line in enumerate(md_lines):
        new_line = line
        for match in re.findall(LINK_REGEX, line):
            if len(match) == 2:
                txt, url = match
                if not re.match(ABSOLUTE_URL_REGEX, url):
                    # url is a relative url
                    new_url = url_prefix + url
                    new_line = new_line.replace(f"[{txt}]({url})", f"[{txt}]({new_url})")
        md_lines[idx] = new_line

    return "\n".join(md_lines)


def _process_md(text: str, url: Optional[str] = None) -> str:
    """
    Process markdown text to yield better text chunks when text is split

    1. Remove excessive newlines before/after code blocks
    2. Use mdformat for general cleanup
    3. Remove large code blocks
    4. Convert tables into condensed format
    5. Convert relative URL links into absolute URL links
    """
    # strip empty newlines before end of code blocks
    md = re.sub(r"\n\n+```", "\n```", text)
    # strip empty newlines after the start of a code block
    md = re.sub(r"```\n\n+", "```\n", md)
    # use opinionated formatter
    md = mdformat.text(md)

    md = _remove_large_md_code_blocks(md)
    md = _convert_md_tables(md)
    if url:
        try:
            md = _convert_relative_links(md, url)
        except Exception:
            log.exception("hit unexpected error while converting relative links")

    return md


def _mkdocs_to_md(md_content: str) -> str:
    """
    Parse a .html page that has been composed with mkdocs

    Converts the page back into md using html2text and addresses formatting issues that
    are commonly found after the conversion.
    """
    # remove "Edit this page" button
    edit_button = md_content.find("a", title="Edit this page")
    if edit_button:
        edit_button.decompose()

    # remove line numbers from code blocks
    linenos_columns = md_content.find_all("td", class_="linenos")
    for linenos_column in linenos_columns:
        linenos_column.decompose()

    h = html2text.HTML2Text()
    h.ignore_images = True
    h.mark_code = True
    h.body_width = 0
    h.ignore_emphasis = True
    h.wrap_links = False
    h.ignore_tables = True

    html2text_output = h.handle(str(md_content))

    md_lines = []
    in_code_block = False

    for line in html2text_output.split("\n"):
        # remove non printable chars (like paragraph markers)
        line = "".join(filter(lambda char: char in string.printable, line))

        # remove trailing "#" from header lines
        if re.match(r"#+ \S+", line):
            line = line.rstrip("\\\\#")

        # replace html2text code block start/end with standard md
        if "[code]" in line:
            in_code_block = True
            line = line.replace("[code]", "```")
        elif "[/code]" in line:
            in_code_block = False
            line = line.replace("[/code]", "```")
        if in_code_block:
            # also fix indent, html2text indents all code content by 4 spaces
            if line.startswith("```"):
                # start of code block and the txt may be on the same line...
                #   eg: ```    <text>
                line = f"```\n{line[8:]}"
            else:
                line = line[4:]
        md_lines.append(line)

    md = "\n".join(md_lines)
    return md


class File:
    def __init__(
        self,
        source: str,
        full_path: str,
        active: bool = True,
        pending_removal: bool = False,
        content: Optional[str] = "",
        hash: Optional[str] = "",
        citation_url: Optional[str] = None,
        **kwargs,
    ):
        self.source = source
        self.full_path = full_path
        self.content = content
        self.active = active
        self.pending_removal = pending_removal
        self.hash = hash
        self.citation_url = citation_url

    def validate(self):
        validate_file_path(self.full_path)
        validate_file_type(self.full_path)
        validate_source(self.source)

    @property
    def metadata(self) -> dict:
        metadata = {
            "source": self.source,
            "full_path": self.full_path,
            "citation_url": self.citation_url,
            "active": self.active,
            "pending_removal": self.pending_removal,
            "hash": self.hash,
        }
        return {key: str(val) for key, val in metadata.items()}

    @property
    def display_name(self) -> str:
        return f"{self.source}:{self.full_path}"

    def __str__(self) -> str:
        return self.display_name

    def extract_text(self):
        if self.full_path.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(StringIO(self.content))

            text_content = ""
            # iterate through each page in the PDF and extract text
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text()

            return text_content

        if self.full_path.endswith(".html"):
            soup = BeautifulSoup(self.content, "lxml")
            # look for 'md-content' in the page, this ignores nav/header/footer/etc.
            md_content = soup.find("div", class_="md-content")
            if md_content:
                # assume this is an mkdocs html page
                md = _mkdocs_to_md(md_content)
                return _process_md(md, url=self.citation_url)
            else:
                return self.content

        if self.full_path.endswith(".md"):
            return _process_md(self.content, url=self.citation_url)

        if self.full_path.endswith(".adoc"):
            return _adoc_to_md(os.path.basename(self.full_path), self.content)

        if self.full_path.endswith(".txt") or self.full_path.endswith(".rst"):
            return self.content

        log.error("cannot extract text for unsupported file type: %s", self.full_path)
        return ""
