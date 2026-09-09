"""Internal JSON-lines protocol; native diagnostics go to stderr."""
import array
import json
import os
import sys

from .nvidia import NvidiaRecognizer


def main():
    # The standalone CLI uses console Python. OpenWhisper's windowed application
    # retains its own tested standard-handle restoration entry point.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    engine = None
    try:
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                if request['op'] == 'load':
                    if engine is not None:
                        engine.close()
                    engine = None
                    engine = NvidiaRecognizer(request['runtime'], request['model_path'], request['device'])
                    result = {'device': request['device']}
                elif request['op'] == 'transcribe':
                    if engine is None:
                        raise RuntimeError('No model loaded')
                    samples = array.array('f')
                    with open(request['audio_path'], 'rb') as stream:
                        samples.frombytes(stream.read())
                    result = engine.transcribe(samples, None)
                else:
                    raise ValueError('Unknown worker operation')
                response = {'id': request['id'], 'result': result}
            except Exception as exc:
                response = {'id': request.get('id'), 'error': str(exc)}
            protocol.write(json.dumps(response, ensure_ascii=True) + '\n')
    finally:
        if engine is not None:
            engine.close()


if __name__ == '__main__':
    main()
