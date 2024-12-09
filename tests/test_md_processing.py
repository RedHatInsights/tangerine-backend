import pytest
from connectors.db.file import _convert_relative_links


@pytest.mark.parametrize(
    "url",
    ("http://baseurl.com/path/to/docs/index.html", "http://baseurl.com/path/to/docs/"),
    ids=("file_url", "folder_url"),
)
def test_link_conversion(url):
    test_txt = """
        This is a markdown file. Some links are [absolute links](http://something.com)

        Meanwhile, others are [relative](somewhere.html) [links](../back/somewhere/else.html)

        [relative](#relative-header) links should be converted and [absolute](https://absolute.com/index.html) should not.
    """

    expected_txt = """
        This is a markdown file. Some links are [absolute links](http://something.com)

        Meanwhile, others are [relative](http://baseurl.com/path/to/docs/somewhere.html) [links](http://baseurl.com/path/to/docs/../back/somewhere/else.html)

        [relative](http://baseurl.com/path/to/docs/#relative-header) links should be converted and [absolute](https://absolute.com/index.html) should not.
    """

    assert _convert_relative_links(test_txt, url) == expected_txt


def test_link_conversion_bad_links():
    test_txt = """
        This is a [bad]() link

        This is an [](invalid) link

        This is a totally []() invalid link
    """
    base_url = "http://baseurl.com/path/to/docs"

    # text should be unmodified
    assert _convert_relative_links(test_txt, base_url) == test_txt
