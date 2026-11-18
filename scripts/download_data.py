import hashlib
import json
import urllib.request
from datetime import datetime, timezone

from common import ROOT


def main():
    urls = {'GuardianReportsAll.csv': 'https://raw.githubusercontent.com/RyanBeal7/GuardianPreviewData/master/GuardianReportsAll.csv'}
    urls.update({f'raw_results/E0_{s}.csv': f'https://www.football-data.co.uk/mmz4281/{s}/E0.csv'
                 for s in ['1314', '1415', '1516', '1617', '1718']})
    manifest = []
    path = ROOT / 'data/source_manifest.json'
    existing = {row['path']: row for row in json.loads(path.read_text())} if path.exists() else {}
    for name, url in urls.items():
        dest = ROOT / 'data/raw' / name
        relative = str(dest.relative_to(ROOT))
        if dest.exists():
            content = dest.read_bytes()
            if relative not in existing or hashlib.sha256(content).hexdigest() != existing[relative]['sha256']:
                raise ValueError(f'Existing file lacks matching download provenance: {dest}')
            manifest.append(existing[relative])
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=45) as response:
            content = response.read()
        dest.write_bytes(content)
        manifest.append({'path': relative, 'url': url, 'sha256': hashlib.sha256(content).hexdigest(),
                         'bytes': len(content), 'retrieved_utc': datetime.now(timezone.utc).isoformat()})
        path.write_text(json.dumps(manifest, indent=2) + '\n')
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Source data ready: {ROOT / "data/raw"}')


if __name__ == '__main__':
    main()
