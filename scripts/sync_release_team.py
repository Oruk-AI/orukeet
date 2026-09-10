"""Carry report authors and marks into release pages.

Requires PyYAML and Pillow; regenerating image assets also requires CairoSVG.
"""
from pathlib import Path
import argparse
import hashlib
import html
import io
import json
import math
import os
import re

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
START, END = '<!-- orukeet-team:start -->', '<!-- orukeet-team:end -->'
BRAND_START, BRAND_END = '<!-- orukeet-brand:start -->', '<!-- orukeet-brand:end -->'
AFFILIATIONS = [
    ('Oruk AI', 'oruk', 'docs/assets/team/oruk-primary.png', 184),
    ('Stanford University', 'stanford', 'report/assets/affiliations/stanford.png', 144),
    ('University of Cambridge', 'cambridge', 'report/assets/affiliations/cambridge.svg', 144),
    ('OpenWhispr', 'openwhispr', 'report/assets/affiliations/openwhispr.svg', 40),
    ('Hoid', 'hoid', 'report/assets/affiliations/hoid.svg', 76),
]
# Destination paths determine relative image URLs in the staged public copies.
PAGES = {
    'README.md': ('README.md', False),
    'MODEL_CARD.md': ('MODEL_CARD.md', False),
    'hub/README.md': ('README.md', True),
    'hub/technical-report.md': ('docs/technical-report.md', True),
    'launch/publication/github-README.md': ('README.md', False),
    'launch/publication/github-MODEL_CARD.md': ('MODEL_CARD.md', False),
    'launch/publication/huggingface-README.md': ('README.md', True),
    'docs/technical-report.md': ('docs/technical-report.md', False),
    'report/README.md': ('report/README.md', False),
    'launch/README.md': ('launch/README.md', False),
    'launch/CURRENT_RELEASE.md': ('launch/CURRENT_RELEASE.md', False),
    'launch/announcements.md': ('launch/announcements.md', False),
    'launch/press-kit/README.md': ('launch/press-kit/README.md', False),
    'launch/press-kit/fact-sheet.md': ('launch/press-kit/fact-sheet.md', False),
    'launch/press-kit/launch-article.md': ('launch/press-kit/launch-article.md', False),
    'demos/multilingual/README.md': ('demos/multilingual/README.md', False),
    'integrations/openwhispr/MODEL_CARD.md': ('integrations/openwhispr/MODEL_CARD.md', False),
}
REQUIRED_PAGES = {'README.md', 'MODEL_CARD.md', 'docs/technical-report.md', 'report/README.md'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def team():
    record = yaml.safe_load((ROOT / 'CITATION.cff').read_text())
    authors = []
    for author in record['authors']:
        name = author['given-names'] + ' ' + author['family-names']
        affiliations = author['affiliation'].split('; ')
        numbers = [1 + [a[0] for a in AFFILIATIONS].index(a) for a in affiliations]
        authors.append({'name': name, 'affiliations': numbers})
    paper = (ROOT / 'report/paper.tex').read_text()
    paper = paper.replace(r'B\"u\c{s}ra Mar\c{s}an', 'Büşra Marşan')
    paper_authors = re.findall(r'([A-Z][A-Za-züşçŞÇğĞöÖıİéèê ]+)\\textsuperscript\{([\d,]+)\}', paper.split(r'\hypersetup')[0])
    parsed = [{'name': name.strip(), 'affiliations': list(map(int, numbers.split(',')))} for name, numbers in paper_authors]
    assert parsed == authors, 'The release team must match the report, including order and affiliations'
    assert len(authors) == 9 and authors[-1]['name'] == 'Calbert Graham'
    return authors


def asset_url(destination, hub, slug):
    if hub:
        return ('../' if str(destination).startswith('docs/') else '') + f'affiliations/{slug}.png'
    return Path(os.path.relpath(ROOT / f'docs/assets/team/{slug}.png', (ROOT / destination).parent)).as_posix()


def blocks(destination, hub=False):
    authors = team()
    brand = f'{BRAND_START}\n<p><img src="{asset_url(destination, hub, "oruk")}" alt="Oruk AI" width="184"></p>\n{BRAND_END}'
    lines = []
    for i in range(0, len(authors), 3):
        lines.append(' · '.join(html.escape(a['name']) + '<sup>' + ','.join(map(str, a['affiliations'])) + '</sup>' for a in authors[i:i + 3]))
    byline = '<p>\n' + '<br>\n'.join(lines) + '\n</p>'
    cells = []
    for number, (name, slug, _, width) in enumerate(AFFILIATIONS[1:], 2):
        cells.append(f'<td align="center" valign="middle"><img src="{asset_url(destination, hub, slug)}" alt="{name}" width="{width}"><br><sup>{number}</sup> {name}</td>')
    affiliations = '<p><strong><sup>1</sup> Oruk AI</strong></p>\n<table>\n<tr>\n' + '\n'.join(cells) + '\n</tr>\n</table>'
    return brand, f'{START}\n{byline}\n\n{affiliations}\n{END}'


def stripped(content):
    for start, end in [(START, END), (BRAND_START, BRAND_END)]:
        content = re.sub(re.escape(start) + r'.*?' + re.escape(end) + r'\n*', '', content, flags=re.S)
    return content


def render_page(source, destination, hub=False):
    content = stripped(source)
    # Replace the preceding single-logo treatment and prose-only report byline.
    content = re.sub(r'<img src="[^"]*oruk-lockup\.png" alt="Oruk" width="140">\n*', '', content)
    content = re.sub(r'\nNathan Roll¹[^\n]*\n\n¹ Oruk AI[^\n]*\n', '\n', content)
    match = re.search(r'^# .+$', content, flags=re.M)
    assert match, destination
    brand, byline = blocks(destination, hub)
    return content[:match.start()] + brand + '\n\n' + match.group() + '\n\n' + byline + '\n\n' + content[match.end():].lstrip('\n')


def build_assets():
    import cairosvg
    folder = ROOT / 'docs/assets/team'
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for number, (name, slug, source, width) in enumerate(AFFILIATIONS, 1):
        path = ROOT / source
        if path.suffix == '.svg':
            picture = Image.open(io.BytesIO(cairosvg.svg2png(url=str(path), output_width=1200))).convert('RGBA')
        else:
            picture = Image.open(path).convert('RGBA')
        bbox = picture.getchannel('A').getbbox()
        assert bbox
        picture = picture.crop(bbox)
        picture.thumbnail((720, 280), Image.Resampling.LANCZOS)
        # The September 2026 Oruk guide requires at least half a Signal tile
        # of clear space. Half the full mark height is a conservative bound.
        padding = math.ceil(picture.height / 2) if slug == 'oruk' else 16
        plate = Image.new('RGBA', (picture.width + padding * 2, picture.height + padding * 2), 'white')
        plate.alpha_composite(picture, (padding, padding))
        output = folder / (slug + '.png')
        plate.convert('RGB').save(output, optimize=True)
        records.append(dict(id=number, name=name, slug=slug, primary=number == 1,
                            source=source, source_sha256=sha(path),
                            web_asset=output.relative_to(ROOT).as_posix(),
                            web_asset_sha256=sha(output), display_width=width))
    return records


def validate():
    manifest = json.loads((ROOT / 'docs/team.json').read_text())
    assert manifest['authors'] == team()
    assert manifest['primary_organization'] == 'Oruk AI'
    for affiliation in manifest['affiliations']:
        assert sha(ROOT / affiliation['source']) == affiliation['source_sha256']
        assert sha(ROOT / affiliation['web_asset']) == affiliation['web_asset_sha256']
    pages = 0
    for source, (destination, hub) in PAGES.items():
        if not (ROOT / source).exists():
            assert source not in REQUIRED_PAGES, source
            continue
        content = (ROOT / source).read_text()
        brand, byline = blocks(destination, hub)
        assert content.count(brand) == content.count(byline) == 1, source
        assert content.index(brand) < content.index(byline), source
        assert len(re.findall(r'<img ', brand + byline)) == 5
        pages += 1
    return dict(status='passed', authors=9, affiliations=5, pages=pages,
                primary_organization='Oruk AI', publication_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if not args.check:
        affiliations = build_assets()
        manifest = dict(primary_organization='Oruk AI', authors=team(), affiliations=affiliations,
                        source_report='report/paper.tex', citation='CITATION.cff',
                        presentation='Oruk leads each page; the remaining marks label author affiliations in report order.')
        (ROOT / 'docs/team.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        for source, (destination, hub) in PAGES.items():
            path = ROOT / source
            if path.exists():
                path.write_text(render_page(path.read_text(), destination, hub))
    print(json.dumps(validate()))


if __name__ == '__main__':
    main()
