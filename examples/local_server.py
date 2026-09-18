"""A bounded, local OpenAI-compatible transcription example. See local-server.md."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hmac
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import threading
import wave

import av
from python_multipart.exceptions import MultipartParseError
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartException
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

LOG = logging.getLogger("orukeet.local_server")
BODY_LIMIT_ERROR = "Upload exceeds the configured byte limit"
# Deliberately exclude playlists and image/document/network demuxers.
AUDIO_FORMATS = "wav,mp3,flac,ogg,mov,matroska,webm,aac"


class InvalidAudio(ValueError):
    pass


def normalize_audio(source: Path, destination: Path, max_seconds: float) -> None:
    """Decode finite audio to a bounded, trusted mono PCM file, without URLs."""
    count = 0
    limit = int(max_seconds * 16000)
    try:
        # A file object, restricted demuxers and an empty protocol allowlist prevent
        # uploaded playlists or container references from fetching other resources.
        with source.open("rb") as raw, av.open(
            raw, options={"format_whitelist": AUDIO_FORMATS, "protocol_whitelist": ""}
        ) as container, wave.open(str(destination), "wb") as output:
            if not container.streams.audio:
                raise InvalidAudio("The upload contains no audio stream")
            output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

            def frames():
                for frame in container.decode(audio=0):
                    frame.pts = None
                    yield from resampler.resample(frame)
                yield from resampler.resample(None)

            for frame in frames():
                count += frame.samples
                if count > limit:
                    raise InvalidAudio(f"Audio exceeds the {max_seconds:g}-second limit")
                output.writeframesraw(frame.to_ndarray().tobytes())
        if count == 0:
            raise InvalidAudio("The upload contains no decoded audio")
    except (av.FFmpegError, OSError, ValueError) as error:
        if isinstance(error, InvalidAudio):
            raise
        raise InvalidAudio("Cannot decode this audio; use WAV, FLAC, MP3, OGG, M4A or WebM") from error


def load_model(installation: Path):
    from orukeet import Orukeet

    config = json.loads(installation.read_text(encoding="utf-8-sig"))
    LOG.info("Loading one Orukeet model for server process %s", os.getpid())
    model = Orukeet(config["model"], config["runtime"], device=config["device"])
    LOG.info("Orukeet ready; persistent native worker PID %s", model._worker.process.pid)
    return model


def create_app(
    installation: Path,
    *,
    max_upload_bytes: int = 25 * 1024 * 1024,
    max_seconds: float = 300,
    max_requests: int = 2,
    api_key: str | None = None,
    model_factory=load_model,
) -> Starlette:
    """Keep exactly one model and at most max_requests admitted uploads/jobs."""
    if max_upload_bytes <= 0 or not math.isfinite(max_seconds) or max_seconds <= 0 or max_requests <= 0:
        raise ValueError("Upload, duration and request limits must be positive")
    slots = threading.BoundedSemaphore(max_requests)

    async def wait_for_job(job):
        future = asyncio.wrap_future(job)
        future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.shield(future)

    @asynccontextmanager
    async def lifespan(app):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="orukeet")
        load_job = executor.submit(model_factory, installation)
        try:
            model = await wait_for_job(load_job)
            app.state.model = model
            app.state.executor = executor
            yield
        finally:
            def close_loaded_model():
                # Queue cleanup behind loading/inference even when startup or an
                # HTTP caller is cancelled before receiving the model/result.
                try:
                    loaded = load_job.result()
                except Exception:
                    return
                loaded.close()

            try:
                await wait_for_job(executor.submit(close_loaded_model))
            finally:
                await asyncio.to_thread(executor.shutdown, wait=True)

    def error(message, status=400, *, headers=None):
        return JSONResponse(
            {"error": {"message": message, "type": "invalid_request_error" if status < 500 else "server_error"}},
            status_code=status, headers=headers,
        )

    async def health(_request):
        return JSONResponse({"status": "ready", "model": "orukeet"})

    async def transcribe(request: Request):
        if api_key and not hmac.compare_digest(
            request.headers.get("authorization", "").encode("utf-8"), f"Bearer {api_key}".encode("utf-8")
        ):
            return error("Invalid API key", 401)
        # No browser-origin requests; this example is for local clients, without CORS.
        if request.headers.get("origin"):
            return error("Browser-origin requests are not supported", 403)
        if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
            return error("Use multipart/form-data with file and model fields", 415)
        if not slots.acquire(blocking=False):
            return error("The transcription worker is busy; retry later", 429, headers={"Retry-After": "1"})
        owns_slot = True
        upload_dir = None
        try:
            received = 0

            async def limited_receive():
                nonlocal received
                message = await request.receive()
                if message["type"] == "http.request":
                    received += len(message.get("body", b""))
                    if received > max_upload_bytes:
                        # The multipart parser closes partially spooled files on this exception.
                        raise MultiPartException(BODY_LIMIT_ERROR)
                return message

            bounded_request = Request(request.scope, limited_receive)
            async with bounded_request.form(max_files=1, max_fields=8, max_part_size=4096) as form:
                pairs = form.multi_items()
                if len({key for key, _value in pairs}) != len(pairs):
                    return error("Duplicate form fields are not supported")
                unknown = set(form) - {"file", "model", "response_format", "language", "prompt", "temperature", "temperature_inc"}
                if unknown:
                    return error("Unsupported form field: " + sorted(unknown)[0])
                if form.get("model") not in ("orukeet", "whisper-1"):
                    return error("model must be orukeet (whisper-1 is a compatibility alias)")
                if form.get("language", "auto") not in ("", "auto"):
                    return error("Orukeet detects language automatically; use language=auto or omit it")
                if form.get("prompt", "") != "":
                    return error("Prompt conditioning is not supported")
                if form.get("temperature", "0") not in ("", "0", "0.0"):
                    return error("Only temperature=0 is supported")
                if form.get("temperature_inc", "0.2") not in ("", "0", "0.0", "0.2"):
                    return error("temperature_inc is only accepted as a legacy no-op (0 or 0.2)")
                response_format = form.get("response_format", "json")
                if response_format not in ("json", "text"):
                    return error("response_format must be json or text")
                upload = form.get("file")
                if not isinstance(upload, UploadFile):
                    return error("file must be an uploaded audio file")
                upload_dir = tempfile.TemporaryDirectory(prefix="orukeet-upload-")
                source = Path(upload_dir.name) / "upload"
                with source.open("wb") as output:
                    while chunk := await upload.read(1024 * 1024):
                        output.write(chunk)

            worker_dir = upload_dir

            def recognize():
                try:
                    audio = source.with_name("audio.wav")
                    normalize_audio(source, audio, max_seconds)
                    return request.app.state.model.transcribe(audio)["text"]
                finally:
                    worker_dir.cleanup()

            job = request.app.state.executor.submit(recognize)
            # The native job retains its slot/tempfile even if its HTTP caller leaves.
            # Releasing on request cancellation would permit an unbounded work queue.
            job.add_done_callback(lambda _future: slots.release())
            owns_slot = False
            upload_dir = None  # ownership passed to worker_dir
            text = await wait_for_job(job)
            return PlainTextResponse(text) if response_format == "text" else JSONResponse({"text": text})
        except HTTPException as exc:
            return error(str(exc.detail), 413 if exc.detail == BODY_LIMIT_ERROR else exc.status_code)
        except MultipartParseError:
            return error("Malformed multipart upload")
        except InvalidAudio as exc:
            return error(str(exc))
        except Exception:
            LOG.exception("Transcription request failed")
            return error("Transcription failed; see the server log", 500)
        finally:
            if upload_dir is not None:
                upload_dir.cleanup()
            if owns_slot:
                slots.release()

    return Starlette(routes=[Route("/health", health), Route("/v1/audio/transcriptions", transcribe, methods=["POST"])], lifespan=lifespan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installation", type=Path, default=Path("installation.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-upload-mb", type=int, default=25)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--max-requests", type=int, default=2, help="Total active/queued uploads, including decoding")
    args = parser.parse_args()
    key = os.environ.get("ORUKEET_API_KEY")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not key:
        parser.error("Set ORUKEET_API_KEY before binding to a non-loopback address")
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(args.installation, max_upload_bytes=args.max_upload_mb * 1024 * 1024,
                          max_seconds=args.max_seconds, max_requests=args.max_requests, api_key=key),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
