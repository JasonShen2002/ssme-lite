from html.parser import HTMLParser
from importlib.resources import files


def test_four_chapters_preserve_report_panels():
    class Structure(HTMLParser):
        def __init__(self):
            super().__init__()
            self.sections, self.ids, self.parents = [], [], {}
            self.chapter = None

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'section':
                self.chapter = attrs['id']
                self.sections.append(self.chapter)
            if 'id' in attrs:
                self.ids.append(attrs['id'])
                self.parents[attrs['id']] = self.chapter

    template = files('ssme_lite.reporting').joinpath('assets/report.html').read_text(encoding='utf-8')
    parsed = Structure()
    parsed.feed(template)
    assert parsed.sections == ['overview', 'performance', 'landscape', 'space']
    assert len(parsed.ids) == len(set(parsed.ids))
    for panel, chapter in {'setup': 'overview', 'hero-dot': 'overview',
                           'ranking': 'performance', 'metric-table': 'performance',
                           'sampling': 'performance', 'contribution': 'landscape',
                           'correlation': 'landscape', 'diagnostics': 'space',
                           'pca-3d': 'space', 'ambiguous-table': 'space'}.items():
        assert parsed.parents[panel] == chapter
    assert template.count('class="section-index"') == 4
    assert template.count('<details class="data-details" open>') == 2
