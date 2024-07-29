from io import BytesIO

import PyPDF2


def text_extractor(filename, file_content):
    if filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))

        text_content = ""
        # iterate through each page in the PDF and extract text
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text_content += page.extract_text()

        return text_content

    if filename.endswith('.md') or filename.endswith('.txt') or filename.endswith('.rst'):
        text_content = file_content.decode('utf-8')
        return text_content

    return ''
