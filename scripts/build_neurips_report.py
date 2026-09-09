"""Compile the short technical report against the selected checkpoint's evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-pages',type=int,default=5)
    p.add_argument('--allow-draft-pages',action='store_true')
    p.add_argument('--tex-bin',type=Path,help='TeX Live bin directory; compile with pdfLaTeX and BibTeX')
    a=p.parse_args()
    report=ROOT/'report';model=read(report/'model.json')
    subprocess.run([sys.executable,str(ROOT/'evaluation/standard_asr/build_current_report.py')],check=True)
    metrics=read(report/'current-benchmark-validation.json')
    assert metrics['status']=='passed' and metrics['models']['orukeet']==model['sha256']
    audit=read(ROOT/model['freeze_audit'])
    assert audit['status']=='pass' and audit['candidate_sha256']==model['sha256']
    assert audit['gabor_values_match_parent_and_original_functions'] and audit['frozen_gabor_rows_exact']==12288
    assert audit['every_other_parameter_tensor_changed'] and audit['tokenizer_assets_equal']
    archive=read(ROOT/model['archive_receipt'])
    # The archive receipt is independently verified at its immutable private revision.
    assert model.get('archive_hf', model['hf'])['revision'] in json.dumps(archive) and model['sha256'] in json.dumps(archive)
    figures=read(report/'assets/model-figures-source.json')
    assert figures['inputs']['training/gabor_half/fits/fits.npz']==sha(ROOT/'training/gabor_half/fits/fits.npz')
    for name in ['kernel-fits','selection-profile']:
        assert figures['figures'][name]['sha256']['pdf']==sha(report/'assets'/(name+'.pdf'))
    paper=(report/'paper.tex').read_text()
    assert model['sha256'][:12] in paper
    assert len(re.findall(r'\\begin\{figure\}',paper))==2
    assert 'gabormer' not in paper.lower() and 'leaderboard' not in paper.lower()
    build=ROOT/'.tmp/neurips-build';build.mkdir(parents=True,exist_ok=True)
    output=ROOT/'output/pdf';output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,SOURCE_DATE_EPOCH='1788825600',FORCE_SOURCE_DATE='1')
    if a.tex_bin:
        binary=(a.tex_bin/'pdflatex').resolve();bibtex=(a.tex_bin/'bibtex').resolve()
        # Preserve the executable symlink: pdftex selects its format from argv[0].
        binary=a.tex_bin.absolute()/'pdflatex';bibtex=a.tex_bin.absolute()/'bibtex'
        command=[str(binary),'-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-file-line-error','-recorder','-output-directory',str(build),'paper.tex']
        commands=[(command,report),([str(bibtex),'paper'],build),(command,report),(command,report)]
        env.update(BIBINPUTS=str(report)+os.pathsep,PATH=str(a.tex_bin.absolute())+os.pathsep+os.environ['PATH'])
        compiler=subprocess.check_output([str(binary),'--version'],text=True).splitlines()[0]
    else:
        commands=[(['tectonic','--keep-logs','--keep-intermediates','--outdir',str(build),'paper.tex'],report)]
        compiler=subprocess.check_output(['tectonic','--version'],text=True).strip()
    log=[]
    for command,cwd in commands:
        process=subprocess.run(command,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log.append(process.stdout)
        (build/'compile-output.txt').write_text('\n'.join(log))
        if process.returncode:print(process.stdout)
        process.check_returncode()
    pdf=build/'paper.pdf';info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    pages=int(re.search(r'^Pages:\s+(\d+)',info,re.MULTILINE).group(1))
    if pages!=a.expected_pages and not a.allow_draft_pages:
        raise RuntimeError(f'Expected {a.expected_pages} pages including tables and references; got {pages}')
    dest=output/'orukeet-technical-report.pdf';shutil.copyfile(pdf,dest)
    dist=ROOT/'dist';dist.mkdir(exist_ok=True);shutil.copyfile(pdf,dist/dest.name)
    subprocess.run([sys.executable,str(ROOT/'evaluation/standard_asr/build_current_report.py'),'--verify-pdf'],check=True)
    inputs=['report/paper.tex','report/model.json','report/references.bib','report/neurips_2026.sty',
        'report/current-benchmark-values.tex','report/current-standard-table.tex','report/current-domains-table.tex',
        'report/current-benchmark-validation.json','scripts/build_neurips_report.py',
        'evaluation/standard_asr/build_current_report.py','evaluation/standard_asr/run.py',
        'evaluation/standard_asr/rescore.py','evaluation/standard_asr/audit_predictions.py',
        'evaluation/standard_asr/scoring.py','report/assets/model-figures-source.json',
        'report/assets/kernel-fits.pdf','report/assets/selection-profile.pdf','report/assets/oruk-lockup.png',
        'training/gabor_half/fits/fits.npz',model['freeze_audit'],model['archive_receipt']]
    inputs+=list(metrics['inputs_sha256'])
    inputs+=[p.relative_to(ROOT).as_posix() for p in sorted((report/'assets/affiliations').iterdir()) if p.is_file()]
    inputs+=[p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'evaluation/standard_asr/vendor').iterdir()) if p.is_file()]
    receipt=dict(status='compiled-awaiting-visual-review',pages=pages,expected_pages=a.expected_pages,
        pdf=dest.relative_to(ROOT).as_posix(),sha256=sha(dest),compiler=compiler,
        format='Short technical report in the official NeurIPS 2026 preprint style',report_model=model,
        inputs_sha256={name:sha(ROOT/name) for name in sorted(set(inputs))},publication_authorized=model.get('publication_authorized', False))
    previous_path=report/'build-receipt.json'
    if previous_path.exists():
        previous=read(previous_path)
        if previous.get('sha256')==receipt['sha256'] and previous.get('visual_review'):
            receipt.update(visual_review=previous['visual_review'],status='compiled-and-visually-reviewed')
    previous_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(pages=pages,pdf=str(dest),sha256=receipt['sha256'])))

if __name__=='__main__':main()
