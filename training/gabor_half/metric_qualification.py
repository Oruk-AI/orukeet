"""Interpret matched NeMo statistics under the unchanged surgery criteria."""


def qualify(result, development_status, diagnostic_only):
    result = dict(result)
    result['interpretation'] = ('Matched NeMo/bf16 greedy decoding; existing exposed release recordings. '
                                'Not an unseen-data or universal-accuracy claim.')
    failures = []
    if result['primary']['delta_pp'] > 0:
        failures.append('Primary WER exceeds original')
    if result['english']['delta_pp'] > 0:
        failures.append('English macro WER exceeds original')
    for name, metric in result['metrics'].items():
        cap = .5 if name.startswith('language:') else .3 if name.startswith('english:') else None
        if cap is not None and metric['delta_pp'] > cap:
            failures.append(name + ' exceeds regression guardrail')
    result.update(status='pass' if not failures else 'fail', failures=failures,
                  development_status=development_status, diagnostic_only=diagnostic_only,
                  release_qualified=not failures and development_status == 'pass' and not diagnostic_only)
    return result
