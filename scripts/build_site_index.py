"""Regenerate site/data/index.json from all *.json data files in site/data/."""
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'site' / 'data'

items = []
for f in sorted(DATA_DIR.glob('*.json')):
    if f.name == 'index.json':
        continue
    d = json.loads(f.read_text())
    items.append({
        'file': f.name,
        'id': d['id'], 'label': d['label'],
        'dataset': d['dataset'], 'subject': d['subject'],
        'sr_factor': d['sr_factor'], 'n_gaussians': d['n_gaussians'],
        'n_time': d['n_time'], 'sampling_rate': d['sampling_rate'],
    })
out = DATA_DIR / 'index.json'
out.write_text(json.dumps(items, indent=2))
print(f"Wrote {out} with {len(items)} entries", file=sys.stderr)
