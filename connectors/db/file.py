import re
from io import StringIO
from typing import Optional

import PyPDF2


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


class File:
    def __init__(
        self,
        source: str,
        full_path: str,
        active: bool = True,
        pending_removal: bool = False,
        content: Optional[str] = "",
    ):
        self.source = source
        self.full_path = full_path
        self.content = content
        self.active = active
        self.pending_removal = pending_removal

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

        if self.full_path.endswith((".md", ".txt", ".rst", ".html")):
            return self.content

        return ""
