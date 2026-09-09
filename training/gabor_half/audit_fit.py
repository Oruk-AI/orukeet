import argparse
import json
from pathlib import Path
import numpy as np
from fit import sha,reconstruct,SOURCE_SHA
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
r=json.loads((a.directory/'fits.json').read_text());z=np.load(a.directory/'fits.npz')
assert r['source_sha256']==SOURCE_SHA and r['kernels']==24576 and r['selected']==12288
assert sha(a.directory/'fits.npz')==r['fit_archive_sha256']
assert np.isfinite(z['params']).all() and np.isfinite(z['fitted']).all()
np.testing.assert_array_equal(reconstruct(z['params']).astype('float32'),z['fitted'])
params=np.array([x['params'] for x in r['records']]);np.testing.assert_array_equal(params,z['params'])
mask=z['selected'];assert int(mask.sum())==12288
order=np.argsort(z['normalized_sse'],kind='stable');assert mask[order[:12288]].all() and not mask[order[12288:]].any()
counts={}
for row in r['records']:counts[row['name']]=counts.get(row['name'],0)+int(row['selected'])
out={'status':'pass','global_ranking_verified':True,'selected':int(mask.sum()),'per_layer':counts,
 'selected_relative_rms_median':float(np.median(np.sqrt(z['normalized_sse'][mask]))),
 'selected_relative_rms_max':float(np.sqrt(z['normalized_sse'][mask]).max()),
 'selected_energy_weighted_relative_sse':float(((z['original'][mask]-z['fitted'][mask])**2).sum()/(z['original'][mask]**2).sum())}
(a.directory/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
