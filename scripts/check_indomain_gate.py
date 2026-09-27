"""CPU-only independent arithmetic and rejection tests; no dataset inference."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import torch

ROOT = Path('/user/tomcosmo.fischer/u27846/repos/fixation-prediction')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('runner', ROOT/'scripts/run_frozen_indomain.py')
ev = load('evaluator', ROOT/'scripts/eval_synthetic_fair.py')
torch.manual_seed(17)
for grid in [4, 64, 224]:
    p = torch.rand(3, grid, grid)
    p[0] = 1
    p[1] = 1
    p[1, 0, 0] = 4
    p /= p.sum(dim=(-1,-2), keepdim=True)
    lp = p.log()
    xy = torch.tensor([[-1.,-1.],[-1.,-1.],[1.,1.]])
    values = ev.per_fixation_metrics(lp, xy)
    for i in range(3):
        x = int(round(float((xy[i,0]+1)/2*(grid-1))))
        y = int(round(float((xy[i,1]+1)/2*(grid-1))))
        flat = lp[i].flatten().tolist(); pos = y*grid+x
        others = flat[:pos]+flat[pos+1:]
        auc = sum(1 if v < flat[pos] else .5 if v == flat[pos] else 0 for v in others)/len(others)
        assert abs(float(values[2][i])-auc) < 1e-6
        assert abs(float(values[0][i])-(flat[pos]+math.log(grid*grid))/math.log(2)) < 2e-6
    assert abs(float(values[0][0])) < 2e-6
    assert abs(float(values[1][0])) < 1e-5
    assert float(values[2][0]) == .5
print('PASS independent LL/AUC and uniform NSS at 4/64/224')
for native, target in [(64,224),(224,64),(224,224)]:
    lp = torch.rand(2,native,native).log_softmax(-1)
    lp = lp - lp.logsumexp((-1,-2),keepdim=True)
    result = ev.to_grid(lp,target)
    assert torch.allclose(result.exp().sum((-1,-2)),torch.ones(2),atol=1e-6)
    if native==target: assert result is lp
print('PASS probability normalization and no native-grid resampling')

path = ROOT/'runs/synthetic_fair/validation/ours_1k_seed42/metrics.json'
d = json.loads(path.read_text());a=d['arguments']
rec=dict(id=a['name'],family='ours',checkpoint=str(ROOT/a['checkpoint']),
         checkpoint_sha256=d['provenance']['checkpoint_sha256'],selection={'ll':1.579540214586258})
m=dict(splits={'validation':dict(path=a['image_list'],images=5000)},
       files={a['image_list']:d['provenance']['image_list_sha256']},
       parquet=a['parquet_path'],images=a['imagenet_root'],native_ll_tolerance=.005)
runner.check_result(path,rec,m,'validation')
mutations=[
 ('wrong checkpoint',lambda x:x['provenance'].update(checkpoint_sha256='bad')),
 ('wrong split hash',lambda x:x['provenance'].update(image_list_sha256='bad')),
 ('missing image',lambda x:x['results']['224']['per_image'].pop()),
 ('duplicate image',lambda x:x['results']['224']['per_image'][0].update(image=x['results']['224']['per_image'][1]['image'])),
 ('wrong target count',lambda x:x['results']['224']['per_image'][0].update(n_fixations=239)),
 ('nonfinite',lambda x:x['results']['224']['per_image'][0].update(ll=float('nan'))),
 ('wrong mean',lambda x:x['results']['224'].update(ll_uniform_img=99)),
 ('wrong IDs',lambda x:x['arguments'].update(scanpath_ids=list(range(5)))),
 ('AMP final',lambda x:x['arguments'].update(amp=True)),
]
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)/'bad.json'
    for label,mutate in mutations:
        x=copy.deepcopy(d);mutate(x);p.write_text(json.dumps(x))
        try: runner.check_result(p,rec,m,'validation')
        except RuntimeError: print('PASS rejects',label)
        else: raise AssertionError(label)
    p.write_text('{}')
    try:runner.write_new(p,{'replace':True})
    except FileExistsError:print('PASS rejects overwrite')
    else:raise AssertionError('overwrite')
print('ALL CHECKS PASSED')
