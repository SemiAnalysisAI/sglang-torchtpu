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
        log = kc('logs', pod_name, '--tail=200')
        marker = log.rfind('METRICS ')
        if marker < 0:
            raise RuntimeError(f'{pod_name}: no completed evaluation metrics in logs')
        payload = log[marker + len('METRICS '):]
        start = payload.index('{')
        metrics, end = json.JSONDecoder().raw_decode(payload[start:])
        assert metrics['num_examples'] == expected_rows
        assert metrics['aggregate']['error_rate'] == 0
        assert metrics['total_completion_tokens'] > 0
        assert 'Traceback' not in payload[start + end:]
        print(pod_name, json.dumps({'completed':metrics['num_examples'], 'score':metrics['aggregate']['score']}), flush=True)
    print(kc('delete', 'jobset', name, '--wait=true', '--timeout=120s'), flush=True)
print(kc('get', 'pods', '-o', 'wide'), flush=True)
