import logging
import re
import string
from io import StringIO
from typing import Optional

import html2text
import mdformat
import PyPDF2
from bs4 import BeautifulSoup

log = logging.getLogger("tangerine.file")


def validate_file_path(full_path: str) -> None:
    # intentionally more restrictive, matches a "typical" unix path and filename with extension
    file_regex = r"^[\w\-.\/ ]+\/?\.[\w\-. ]+[^.]$"
    if not full_path or not full_path.strip() or not re.match(file_regex, full_path):
        raise ValueError(f"file path must match regex: {file_regex}")


def validate_source(source: str) -> None:
    source_regex = r"^[\w-]+$"
    if not source or not source.strip() or not re.match(source_regex, source):
        raise ValueError(f"source must match regex: {source_regex}")


def validate_file_type(full_path: str) -> None:
    if not any(
        [full_path.endswith(filetype) for filetype in [".txt", ".pdf", ".md", ".rst", ".html"]]
    ):
        raise ValueError("unsupported file type")


def _remove_large_md_code_blocks(text):
    lines = []
    code_lines = []
    in_code_block = False
    for line in text.split("\n"):
        if line.strip() == "```" and not in_code_block:
            in_code_block = True
            code_lines = []
            code_lines.append(line)
        elif line.strip() == "```" and in_code_block:
            code_lines.append(line)
            in_code_block = False
            if len(code_lines) > 9:
                code_lines = ["```", "<large code block, visit documentation to view>", "```"]
            lines.extend(code_lines)
        elif in_code_block:
            code_lines.append(line)
        else:
            lines.append(line)

    return "\n".join(lines)


def _html_to_md(text):
    """
    Parse a .html page that has been composed with mkdocs

    It is assumed that the page contains 'md-content' which was compiled based on .md

    TODO: possibly handle html pages not built with mkdocs
    """
    md = ""
    soup = BeautifulSoup(text, "lxml")
    # extract 'md-content', this ignores nav/header/footer/etc.
    md_content = soup.find("div", class_="md-content")
    if md_content:
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

        # strip empty newlines before end of code blocks
        md = re.sub(r"\n\n+```", "\n```", md)
        # strip empty newlines after the start of a code block
        md = re.sub(r"```\n\n+", "```\n", md)
        # finally, use opinionated formatter
        md = mdformat.text(md)
    else:
        log.error("no 'md-content' div found")

    md = _remove_large_md_code_blocks(md)

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
        **kwargs,
    ):
        self.source = source
        self.full_path = full_path
        self.content = content
        self.active = active
        self.pending_removal = pending_removal
        self.hash = hash

    def validate(self):
        validate_file_path(self.full_path)
        validate_file_type(self.full_path)
        validate_source(self.source)

    @property
    def metadata(self) -> dict:
        metadata = {
            "source": self.source,
            "full_path": self.full_path,
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

        elif self.full_path.endswith(".html"):
            return _html_to_md(self.content)

        if self.full_path.endswith((".md", ".txt", ".rst", ".html")):
            return self.content

        return ""
