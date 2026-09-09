"""Keep the report citation identical across release cards and GitHub's cite menu."""
import argparse
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
START = '<!-- orukeet-citation:start -->'
END = '<!-- orukeet-citation:end -->'
PAGES = {
    'README.md': 'CITATION.bib',
    'MODEL_CARD.md': 'CITATION.bib',
    'hub/README.md': 'CITATION.bib',
    'launch/publication/github-README.md': 'CITATION.bib',
    'launch/publication/github-MODEL_CARD.md': 'CITATION.bib',
    'launch/publication/huggingface-README.md': 'CITATION.bib',
    'docs/technical-report.md': '../CITATION.bib',
    'report/README.md': '../CITATION.bib',
    'integrations/openwhispr/MODEL_CARD.md': '../../CITATION.bib',
}


def latex(text):
    for character, replacement in {'ü': r'{\"u}', 'ş': r'{\c{s}}'}.items():
        text = text.replace(character, replacement)
    assert text.isascii(), text
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = ROOT/'CITATION.cff'
    record = yaml.safe_load(path.read_text(encoding='utf-8'))
    preferred = dict(type='report', title=record['title'], authors=record['authors'],
                     institution={'name': 'Oruk AI'}, year=2026,
                     url='https://github.com/Oruk-AI/orukeet/blob/main/output/pdf/orukeet-technical-report.pdf')
    if args.check:
        assert record.get('preferred-citation') == preferred
    else:
        record['message'] = 'If you use Orukeet, cite the technical report below and retain the NVIDIA Parakeet attribution.'
        record['preferred-citation'] = preferred
        path.write_text(yaml.safe_dump(record, sort_keys=False, allow_unicode=True, width=100), encoding='utf-8')
    authors = ' and\n            '.join(latex(a['family-names'])+', '+latex(a['given-names']) for a in record['authors'])
    bib = ('@techreport{roll2026orukeet,\n'
           '  title = {{Orukeet}: Multilingual {ASR} with Frozen {Gabor} Kernels},\n'
           f'  author = {{{authors}}},\n'
           '  institution = {Oruk AI},\n'
           '  year = {2026},\n'
           '  type = {Technical report},\n'
           f'  url = {{{preferred["url"]}}}\n'
           '}\n')
    if args.check:
        assert (ROOT/'CITATION.bib').read_text(encoding='utf-8') == bib
    else:
        (ROOT/'CITATION.bib').write_text(bib, encoding='utf-8')
    checked = 0
    for name, target in PAGES.items():
        path = ROOT/name
        if not path.exists() and name not in {'README.md','MODEL_CARD.md','docs/technical-report.md','report/README.md'}:
            continue
        checked += 1
        content = path.read_text(encoding='utf-8')
        block = START+'\n## Citation\n\n```bibtex\n'+bib+'```\n\n[Download BibTeX]('+target+') · [Citation metadata]('+target.replace('.bib', '.cff')+')\n'+END
        if args.check:
            assert block in content and content.count(START) == 1, name
        else:
            if START in content:
                content = re.sub(re.escape(START)+r'.*?'+re.escape(END), lambda _: block, content, flags=re.S)
            else:
                content = content.rstrip()+'\n\n'+block+'\n'
            path.write_text(content, encoding='utf-8')
    print(json.dumps(dict(status='passed', authors=len(record['authors']), pages=checked, citation='roll2026orukeet')))


if __name__ == '__main__':
    main()
