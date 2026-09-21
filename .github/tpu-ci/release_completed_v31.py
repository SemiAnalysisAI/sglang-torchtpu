"""Release only the four completed, locally archived September 20 V3.1 runs."""
import json
import subprocess


def kc(*args):
    return subprocess.check_output(['kubectl', '-n', 'tpu-ci', *args], text=True)


print(kc('get', 'pods', '-o', 'wide'), flush=True)
for repeat in range(2, 6):
    name = f'v31-upstream-repeat{repeat}-20260920'
    pods = json.loads(kc('get', 'pods', '-l', f'jobset.sigs.k8s.io/jobset-name={name}', '-o', 'json'))['items']
    if not pods:
        print(f'{name}: no pods; nothing to release', flush=True)
        continue
    expected_hosts = 3 if repeat == 5 else 2
    expected_rows = 10 if repeat == 5 else 15
    if len(pods) != expected_hosts:
        raise RuntimeError(f'{name}: unexpected pod count')
    for pod in pods:
        pod_name = pod['metadata']['name']
        if pod['status']['phase'] == 'Succeeded':
            continue
        if pod['status']['phase'] != 'Running':
            raise RuntimeError(f'{pod_name}: unexpected phase')
        check = '''import json
from pathlib import Path
p=Path('/cache/upstream-run')
assert (p/'complete.json').is_file(), 'Evaluation not marked complete'
files=list((p/'results').glob('*/metrics.json'))
assert len(files)==1
m=json.loads(files[0].read_text())
assert m['num_examples']==EXPECTED
assert m['aggregate']['error_rate']==0
print(json.dumps({'completed': m['num_examples'], 'score':m['aggregate']['score']}))
'''.replace('EXPECTED', str(expected_rows))
        print(pod_name, kc('exec', pod_name, '--', 'python3', '-c', check), flush=True)
    print(kc('delete', 'jobset', name, '--wait=true', '--timeout=120s'), flush=True)
print(kc('get', 'pods', '-o', 'wide'), flush=True)
