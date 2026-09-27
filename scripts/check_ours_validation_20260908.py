"""Additional release gate: reproduce frozen, rounded prior validation scores."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'docs/report_tables/transfer_test_manifest_ours_20260908.json'
OUTPUTS=ROOT/'runs/frozen_transfer_ours_20260908'
EXPECTED=ROOT/'docs/report_tables/ours_validation_expected_20260908.json'


def check():
    from run_frozen_transfer import verify_validation
    m=json.loads(MANIFEST.read_text())
    mh=hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    verify_validation(m,mh,OUTPUTS)
    expected=json.loads(EXPECTED.read_text())
    assert expected['manifest_sha256']==mh
    assert set(expected['scores'])=={r['id'] for r in m['records']}
    assert len(m['records'])==16 and all(r['family']=='ours' for r in m['records'])
    for r in m['records']:
        met=json.loads((OUTPUTS/'validation'/r['id']/'metrics.json').read_text())['results']['224']
        for key,old in expected['scores'][r['id']].items():
            value=float(met[key])
            assert abs(value-old)<=0.00051,(r['id'],key,value,old)
    print('RELEASE GATE PASSED: all 16 ours runs reproduce prior LL/NSS/AUC within printed rounding',flush=True)


if __name__=='__main__':check()
