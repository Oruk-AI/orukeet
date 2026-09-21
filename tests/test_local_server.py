"""The HTTP example needs its separate requirements; no model downloads in tests."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import io
from pathlib import Path
import threading
import wave

import pytest

pytest.importorskip("starlette")
pytest.importorskip("python_multipart")
httpx = pytest.importorskip("httpx")
from starlette.testclient import TestClient

spec = importlib.util.spec_from_file_location("local_server", Path(__file__).parents[1] / "examples/local_server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def wav(seconds=.05, channels=2):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setparams((channels, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x10\x00" * int(seconds * 16000) * channels)
    return output.getvalue()


class Model:
    def __init__(self):
        self.calls = []
        self.closed = False

    def transcribe(self, path):
        with wave.open(str(path)) as audio:
            assert audio.getnchannels() == 1
            assert audio.getframerate() == 16000
        self.calls.append(path)
        return {"text": "Bonjour, monde."}

    def close(self):
        self.closed = True


def app_for(model, **kwargs):
    return server.create_app(Path("unused.json"), model_factory=lambda _: model, **kwargs)


def post(client, data=None, audio=None):
    return client.post("/v1/audio/transcriptions", data={"model": "orukeet", **(data or {})},
                       files={"file": ("../../recording.wav", wav() if audio is None else audio, "audio/wav")})


def test_persistent_model_stereo_json_text_and_cleanup():
    model = Model()
    loads = []
    def load(path):
        loads.append(path)
        return model
    app = server.create_app(Path("receipt.json"), model_factory=load)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ready"
        assert post(client, {"language": "auto", "temperature": "0.0", "temperature_inc": "0.2"}).json() == {"text": "Bonjour, monde."}
        response = post(client, {"model": "whisper-1", "response_format": "text"})
        assert response.status_code == 200 and response.text == "Bonjour, monde."
        assert response.headers["content-type"].startswith("text/plain")
        assert len(loads) == 1 and len(model.calls) == 2
        assert all(not path.parent.exists() for path in model.calls)
    assert model.closed


@pytest.mark.parametrize("data", [{"model": "unknown"}, {"language": "fr"}, {"prompt": "hint"},
                                   {"response_format": "verbose_json"}, {"stream": "true"},
                                   {"temperature": "1"}, {"temperature_inc": "99"}])
def test_unsupported_options_do_not_infer(data):
    model = Model()
    with TestClient(app_for(model)) as client:
        assert post(client, data).status_code == 400
        assert not model.calls


def test_invalid_empty_playlist_and_overlong_audio_do_not_infer():
    model = Model()
    with TestClient(app_for(model, max_seconds=.1)) as client:
        for payload in [b"invalid", b"", b"#EXTM3U\nhttp://127.0.0.1:9999/private.wav\n", wav(.2)]:
            response = post(client, audio=payload)
            assert response.status_code == 400, response.text
        assert not model.calls
        assert post(client).status_code == 200


def test_streamed_upload_limit_missing_fields_and_duplicate_fields():
    model = Model()
    with TestClient(app_for(model, max_upload_bytes=512)) as client:
        assert post(client).status_code == 413
        assert not model.calls
    with TestClient(app_for(model)) as client:
        assert client.post("/v1/audio/transcriptions", json={}).status_code == 415
        assert client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav())}).status_code == 400
        response = client.post("/v1/audio/transcriptions", data={"model": "orukeet"},
                               files=[("file", ("a.wav", wav())), ("file", ("b.wav", wav()))])
        assert response.status_code == 400
        assert not model.calls


def test_authentication_and_browser_origin():
    model = Model()
    with TestClient(app_for(model, api_key="secret")) as client:
        assert post(client).status_code == 401
        client.headers["Authorization"] = "Bearer secret"
        assert post(client).status_code == 200
        client.headers["Origin"] = "https://example.invalid"
        assert post(client).status_code == 403


def test_cancelled_request_retains_slot_until_native_job_finishes():
    async def run():
        started = threading.Event()
        release = threading.Event()
        model = Model()
        original = model.transcribe
        def blocked(path):
            started.set()
            assert release.wait(5)
            return original(path)
        model.transcribe = blocked
        app = app_for(model, max_requests=1)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                async def submit():
                    return await client.post("/v1/audio/transcriptions", data={"model": "orukeet"}, files={"file": ("a.wav", wav())})
                pending = asyncio.create_task(submit())
                try:
                    assert await asyncio.to_thread(started.wait, 3)
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                    assert (await client.get("/health")).status_code == 200
                    assert (await submit()).status_code == 429
                finally:
                    release.set()
                    await asyncio.wrap_future(app.state.executor.submit(lambda: None))
                assert (await submit()).status_code == 200
        assert model.closed
        assert all(not path.parent.exists() for path in model.calls)
    asyncio.run(run())


def test_parallel_requests_serialize_on_one_worker():
    model = Model()
    with TestClient(app_for(model, max_requests=3)) as client, ThreadPoolExecutor(3) as pool:
        replies = list(pool.map(lambda _: post(client), range(3)))
        assert [reply.status_code for reply in replies] == [200, 200, 200]
        assert len(model.calls) == 3


@pytest.mark.parametrize("kwargs", [{"max_seconds": float("nan")}, {"max_seconds": 0}, {"max_requests": 0}, {"max_upload_bytes": 0}])
def test_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        app_for(Model(), **kwargs)


def test_malformed_multipart_and_non_ascii_auth_are_client_errors():
    model = Model()
    with TestClient(app_for(model)) as client:
        response = client.post("/v1/audio/transcriptions", content=b"--wrong\r\n",
                               headers={"Content-Type": "multipart/form-data; boundary=x"})
        assert response.status_code == 400
        assert not model.calls
    with TestClient(app_for(model, api_key="secret")) as client:
        response = client.post("/v1/audio/transcriptions", headers={b"authorization": b"Bearer \xff"})
        assert response.status_code == 401
        assert client.get("/health").status_code == 200


def test_cancelled_startup_closes_eventual_model():
    async def run():
        started = threading.Event()
        release = threading.Event()
        model = Model()
        def factory(_):
            started.set()
            assert release.wait(5)
            return model
        app = server.create_app(Path("unused.json"), model_factory=factory)
        context = app.router.lifespan_context(app)
        startup = asyncio.create_task(context.__aenter__())
        try:
            assert await asyncio.to_thread(started.wait, 3)
            startup.cancel()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await startup
        assert model.closed
    asyncio.run(run())


def test_shutdown_releases_executor_when_model_close_fails():
    model = Model()
    def broken_close():
        raise RuntimeError("close failed")
    model.close = broken_close
    app = app_for(model)
    with pytest.raises(RuntimeError, match="close failed"):
        with TestClient(app):
            pass
    assert app.state.executor._shutdown
