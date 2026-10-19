"""Authenticated release download and SHA verification; no extraction or solves."""
from pathlib import Path
import subprocess,json
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from scripts.verify_repository import digest
def main():
 manifest=json.loads((ROOT/'data/ASSETS.json').read_text());target=ROOT/'downloads';target.mkdir(exist_ok=True)
 for item in manifest['assets']:
  dest=target/item['name']
  if not dest.exists():subprocess.run(['gh','release','download',manifest['tag'],'--repo',manifest['repository'],'--pattern',item['name'],'--dir',str(target)],check=True)
  if digest(dest)!=item['sha256']:raise RuntimeError('Downloaded hash mismatch: '+item['name'])
  print('VERIFIED',item['name'])
if __name__=='__main__':main()
