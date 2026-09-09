"""Build and independently compile the minimal arXiv source package; never submit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from sync_citation import latex


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tex-bin', required=True, type=Path)
    args = parser.parse_args()
    tex = args.tex_bin.absolute()
    report = ROOT/'report'
    receipt = json.loads((report/'build-receipt.json').read_text())
    assert receipt['status'] == 'compiled-and-visually-reviewed'
    assert receipt['sha256'] == sha(ROOT/receipt['pdf'])
    for name, digest in receipt['inputs_sha256'].items():
        assert sha(ROOT/name) == digest, name
    paper = (report/'paper.tex').read_text()
    cff = yaml.safe_load((ROOT/'CITATION.cff').read_text())
    catalog = json.loads((ROOT/'src/orukeet/artifacts.json').read_text())
    assert catalog['files']['source']['sha256'] == receipt['report_model']['sha256']
    assert catalog['revision'] in paper and '/experiments/' not in paper
    sources = {'paper.tex': report/'paper.tex', 'neurips_2026.sty': report/'neurips_2026.sty',
               'references.bib': report/'references.bib',
               'paper.bbl': ROOT/'.tmp/neurips-build/paper.bbl'}
    for path in re.findall(r'\\input\{([^}]+)\}', paper):
        sources[path] = report/path
    for path in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', paper):
        sources[path] = report/path
    assert len(sources) == 14, sources
    assert r'\begin{thebibliography}' in sources['paper.bbl'].read_text()
    files = {name: path.read_bytes() for name, path in sources.items()}
    config = {'process': {'compiler': 'pdflatex'}, 'texlive_version': 2025,
              'sources': [{'filename': 'paper.tex', 'usage': 'toplevel'}]}
    files['00README.json'] = (json.dumps(config, indent=2)+'\n').encode()
    archive = ROOT/'dist/orukeet-arxiv-source.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(files.items()):
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            info = zipfile.ZipInfo(name, (2026, 9, 8, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data)
    scratch = ROOT/'.tmp/arxiv-check'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='arxiv-clean-', dir=scratch) as name:
        clean = Path(name)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and set(z.namelist()) == set(files)
            z.extractall(clean)
        env = dict(os.environ, PATH=str(tex)+os.pathsep+os.environ['PATH'],
                   SOURCE_DATE_EPOCH='1788825600', FORCE_SOURCE_DATE='1')
        logs = []
        command = [str(tex/'pdflatex'), '-no-shell-escape', '-interaction=nonstopmode',
                   '-halt-on-error', '-file-line-error', '-recorder', 'paper.tex']
        # arXiv can use the supplied .bbl without running BibTeX.
        for _ in range(3):
            result = subprocess.run(command, cwd=clean, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            logs.append(result.stdout)
            (scratch/'arxiv-compile.log').write_text('\n'.join(logs))
            result.check_returncode()
        log = (clean/'paper.log').read_text()
        for error in ['undefined references', 'undefined citations', 'Missing character:', 'Overfull']:
            assert error not in log, error
        rendered = subprocess.check_output(['pdftotext', str(clean/'paper.pdf'), '-'], text=True)
        canonical = subprocess.check_output(['pdftotext', str(ROOT/receipt['pdf']), '-'], text=True)
        assert rendered == canonical, 'Clean source compilation must reproduce every printed character'
        info = subprocess.check_output(['pdfinfo', str(clean/'paper.pdf')], text=True)
        fonts = subprocess.check_output(['pdffonts', str(clean/'paper.pdf')], text=True)
        assert 'Type 3' not in fonts
        assert re.search(r'^Pages:\s+5$', info, re.M) and 'JavaScript:      no' in info
        shutil.copyfile(clean/'paper.pdf', scratch/'arxiv-recompiled.pdf')
        clean_sha = sha(clean/'paper.pdf')
    values = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^{}]+)\}', (report/'current-benchmark-values.tex').read_text()))
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', paper, re.S)[1]
    for key in sorted(values, key=len, reverse=True):
        abstract = re.sub(r'\\'+key+r'\b', lambda _: values[key], abstract)
    abstract = abstract.replace(r'\%', '%').replace(r'\ ', ' ').replace('~', ' ')
    abstract = ' '.join(abstract.split())
    assert '\\' not in abstract and abstract.isascii() and len(abstract) <= 1920
    team = json.loads((ROOT/'docs/team.json').read_text())
    authors = ', '.join(latex(a['name'])+' ('+' and '.join(map(str, a['affiliations']))+')' for a in team['authors'])
    authors += '\n('+', '.join(f'({a["id"]}) {a["name"]}' for a in team['affiliations'])+')'
    metadata = dict(status='prepared-not-submitted', publication_authorized=False,
                    title=cff['title'], authors=authors, abstract=abstract,
                    comments='5 pages, 2 figures. Code and model: https://github.com/Oruk-AI/orukeet',
                    primary_category='cs.SD', cross_list_categories=['cs.LG'],
                    proposed_article_license='CC-BY-4.0',
                    license_url='https://creativecommons.org/licenses/by/4.0/',
                    journal_reference=None, doi=None, arxiv_id=None,
                    tex_processor='pdflatex', texlive_version=2025,
                    model_source_sha256=catalog['files']['source']['sha256'])
    out = ROOT/'release/arxiv'
    out.mkdir(parents=True, exist_ok=True)
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
    fields = '# arXiv submission fields\n\nPrepared privately. No submission has been made.\n'
    for key in ['title', 'authors', 'abstract', 'comments']:
        fields += f'\n## {key.capitalize()}\n\n```text\n{metadata[key]}\n```\n'
    fields += '\nPrimary category: `cs.SD` (Sound). Proposed cross-list: `cs.LG` (Machine Learning). Proposed article license: CC BY 4.0. Leave journal reference and DOI empty. An arXiv identifier will be added to the shared citation after assignment.\n'
    (out/'submission-fields.md').write_text(fields)
    manifest = dict(status='passed', publication_authorized=False, checked_at_utc=datetime.now(timezone.utc).isoformat(),
                    archive=archive.relative_to(ROOT).as_posix(), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
                    file_count=len(files), files=[dict(path=n, bytes=len(d), sha256=hashlib.sha256(d).hexdigest()) for n,d in sorted(files.items())],
                    compiler=subprocess.check_output([str(tex/'pdflatex'), '--version'], text=True).splitlines()[0],
                    distribution='TinyTeX v2025.08 (TeX Live 2025); arXiv selects its own TeX Live 2025 package snapshot',
                    canonical_pdf_sha256=receipt['sha256'], clean_pdf_sha256=clean_sha,
                    canonical_pdf_text_match=True, no_type3_fonts=True, pages=5,
                    abstract_characters=len(abstract), metadata_sha256=sha(out/'metadata.json'),
                    compile_log_sha256=sha(scratch/'arxiv-compile.log'),
                    checks=['minimal archive', 'only required TeX, bibliography and figures',
                            'clean pdfLaTeX build without shell escape', 'supplied bibliography resolves all citations',
                            'exact extracted-text match to reviewed PDF', 'all printed benchmark rows reproduced',
                            'title, nine authors and model identity match citation metadata', 'no arXiv ID or DOI invented'])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (out/'README.md').write_text('''# Orukeet arXiv submission package

The source archive is ready for private review: `dist/orukeet-arxiv-source.zip`.
It contains the main TeX file, style, resolved bibliography, numeric tables,
two figures and five affiliation marks. A clean pdfLaTeX build reproduces the
reviewed five-page report. The exact compiler and file hashes are in [the manifest](manifest.json).

[Copyable submission fields](submission-fields.md) include the title, all nine authors,
affiliations and the manuscript's expanded abstract. The proposed categories are
cs.SD with a cs.LG cross-list; the proposed article license is CC BY 4.0.
The code and weights keep their existing MIT and CC BY-SA 4.0 terms.

For submission, upload the source ZIP, select pdfLaTeX and TeX Live 2025, paste
the prepared fields, and inspect arXiv's generated PDF. The submitter confirms
author consent and the article license in arXiv. The account must meet arXiv's
registration and endorsement requirements. Submit only after release approval;
no arXiv upload or submission has been made here. Coordinate the announcement
with the GitHub and Hugging Face release so the paper's links are accessible.

After an identifier is assigned, add its abstract URL and identifier to the
canonical citation, regenerate the cards, and update the report links. Do not
enter a journal reference or DOI until one exists.

Preparation follows the official [TeX instructions](https://info.arxiv.org/help/submit_tex.html),
[supported processors](https://info.arxiv.org/help/faq/texlive.html),
[00README format](https://info.arxiv.org/help/00README.html),
[metadata rules](https://info.arxiv.org/help/prep.html), and
[license choices](https://info.arxiv.org/help/license/index.html).
''')
    print(json.dumps({k:manifest[k] for k in ['status','archive','archive_bytes','file_count','compiler','pages','abstract_characters']}))


if __name__ == '__main__':
    main()
