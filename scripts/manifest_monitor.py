"""Monitor the ingestion_manifest.json and print notifications whenever a 4-month chunk completes."""
import time
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "data" / "ingestion_manifest.json"
LAST_SEEN = {}

def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"years": {}}

if __name__ == '__main__':
    print('Starting manifest monitor; watching for completed 4-month chunks...')
    while True:
        manifest = load_manifest()
        years = manifest.get('years', {})
        for y, info in years.items():
            chunks = info.get('chunks', {})
            for chunk_key, chunk_info in chunks.items():
                key = f"{y}-{chunk_key}"
                status = chunk_info.get('status')
                if status == 'completed' and LAST_SEEN.get(key) != 'completed':
                    print(f"Completed: year={y} chunk={chunk_key} rows={chunk_info.get('rows')} path={chunk_info.get('path')}")
                    LAST_SEEN[key] = 'completed'
                elif status in ('failed','no_data') and LAST_SEEN.get(key) != status:
                    print(f"Chunk update: year={y} chunk={chunk_key} status={status} info={chunk_info}")
                    LAST_SEEN[key] = status
        time.sleep(60)
