"""Verify every fixed listening example, returned transcript and local audio."""
import hashlib
import html
import json
from pathlib import Path
import wave
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = json.loads((ROOT/'receipt.json').read_text())
    catalog = json.loads((ROOT.parents[1]/'src/orukeet/artifacts.json').read_text())
    assert receipt['status'] == 'completed_with_all_outputs_retained'
    assert receipt['model']['sha256'] == catalog['files']['q8']['sha256']
    assert receipt['selection_sha256'] == sha(ROOT/'provenance/selection.json')
    page = (ROOT/'index.html').read_text()
    assert 'Orukeet r3' in page and len(receipt['clips']) == page.count('<audio ') == 3
    clips = []
    for clip in receipt['clips']:
        assert clip['result'] is not None and clip['error'] is None
        for record in [clip['source']['audio'], clip['playback']]:
            assert sha(ROOT/record['local_file']) == record['sha256']
        tsv = clip['source']['tsv']
        assert sha(ROOT/tsv['selected_row_file']) == tsv['selected_row_sha256']
        assert html.escape(clip['source']['reference_raw']) in page
        assert html.escape(clip['result']['text']) in page
        with wave.open(str(ROOT/clip['playback']['local_file'])) as audio:
            assert audio.getnchannels() == 1 and audio.getframerate() == 16000 and audio.getsampwidth() == 2
            duration = audio.getnframes()/audio.getframerate()
            assert abs(duration - clip['playback']['duration_s']) < 1e-10
        clips.append(dict(config=clip['config'], provider_id=clip['source']['id'], duration_s=duration,
                          source_and_playback_hashes_verified=True, html_preserves_reference_and_prediction=True))
    check = dict(status='passed', checked_at_utc=datetime.now(timezone.utc).isoformat(),
                 scope='r3 fixed-example integrity, returned text, source rows and playback PCM verification',
                 model_sha256=receipt['model']['sha256'], receipt_sha256=sha(ROOT/'receipt.json'),
                 index_html_sha256=sha(ROOT/'index.html'), clips=clips,
                 full_tsv_and_card_files_in_demo=False, external_assets_or_scripts=False)
    assert '<script' not in page and '<link ' not in page
    assert all('src="'+clip['playback']['local_file']+'"' in page for clip in receipt['clips'])
    (ROOT/'verification.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps(dict(status='passed', examples=len(clips), model_sha256=receipt['model']['sha256'])))


if __name__ == '__main__':
    main()
