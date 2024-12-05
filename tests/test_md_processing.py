from connectors.db.file import _convert_relative_links

def test_link_conversion():
    test_txt = '''
        This is a markdown file. Some links are [absolute links](http://something.com)

        Meanwhile, others are [relative](somewhere.html) [links](../back/somewhere/else.html)

        [relative](#relative-header) links should be converted and [absolute](https://absolute.com/index.html) should not.
    '''
    base_url = "http://baseurl.com/path/to/docs"

    expected_txt = '''
        This is a markdown file. Some links are [absolute links](http://something.com)

        Meanwhile, others are [relative](http://baseurl.com/path/to/docs/somewhere.html) [links](http://baseurl.com/path/to/docs/../back/somewhere/else.html)

        [relative](http://baseurl.com/path/to/docs/#relative-header) links should be converted and [absolute](https://absolute.com/index.html) should not.
    '''

    assert _convert_relative_links(test_txt, base_url) == expected_txt
