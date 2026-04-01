from __future__ import annotations

import re
import threading
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Optional

from flask import Flask, jsonify, render_template, request, send_from_directory

from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

# ---------- model state ----------

_model_lock = threading.Lock()
_model: Optional[AudioGen] = None


def get_model() -> AudioGen:
    global _model
    with _model_lock:
        if _model is None:
            _model = AudioGen.get_pretrained("facebook/audiogen-medium")
        return _model


# ---------- jobs ----------

@dataclass
class Job:
    id: str
    prompt: str
    duration: float
    name_prefix: str
    status: str
    stage: str
    progress: int
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    output_file: Optional[str] = None
    error: Optional[str] = None


jobs: dict[str, Job] = {}
job_queue: Queue[str] = Queue()
jobs_lock = threading.Lock()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "sound"


def set_job(job_id: str, **updates) -> None:
    with jobs_lock:
        job = jobs[job_id]
        for key, value in updates.items():
            setattr(job, key, value)


def create_filename(prefix: str, prompt: str, job_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    left = slugify(prefix)[:40] if prefix else ""
    right = slugify(prompt)[:50]
    parts = [p for p in [ts, left, right, job_id[:8]] if p]
    return "_".join(parts) + ".wav"


def list_outputs() -> list[dict]:
    files = []
    for path in sorted(OUTPUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        files.append({
            "name": path.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "url": f"/audio/{path.name}",
        })
    return files


def worker() -> None:
    while True:
        job_id = job_queue.get()
        try:
            with jobs_lock:
                job = jobs[job_id]

            set_job(job_id, status="running", stage="loading model", progress=10, started_at=now_iso())

            model = get_model()

            set_job(job_id, stage="preparing generation", progress=25)
            model.set_generation_params(duration=job.duration)

            set_job(job_id, stage="generating audio", progress=70)
            wav = model.generate([job.prompt])

            filename = create_filename(job.name_prefix, job.prompt, job.id)
            stem = filename[:-4]

            set_job(job_id, stage="writing wav", progress=92)
            audio_write(
                str(OUTPUT_DIR / stem),
                wav[0].cpu(),
                model.sample_rate,
                strategy="loudness",
                loudness_compressor=True,
            )

            set_job(
                job_id,
                status="done",
                stage="finished",
                progress=100,
                finished_at=now_iso(),
                output_file=filename,
            )
        except Exception as exc:
            set_job(
                job_id,
                status="error",
                stage="failed",
                progress=100,
                finished_at=now_iso(),
                error=f"{type(exc).__name__}: {exc}",
            )
            traceback.print_exc()
        finally:
            job_queue.task_done()


threading.Thread(target=worker, daemon=True).start()


# ---------- routes ----------

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/generate")
def api_generate():
    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "").strip()
    duration_raw = data.get("duration")
    name_prefix = (data.get("name_prefix") or "").strip()

    if not prompt:
        return jsonify({"ok": False, "error": "Prompt is required."}), 400

    try:
        duration = float(duration_raw) if duration_raw not in (None, "") else 6.0
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Duration must be a number."}), 400

    if duration <= 0:
        return jsonify({"ok": False, "error": "Duration must be greater than 0."}), 400

    job_id = datetime.now().strftime("%H%M%S%f")
    job = Job(
        id=job_id,
        prompt=prompt,
        duration=duration,
        name_prefix=name_prefix,
        status="queued",
        stage="waiting in queue",
        progress=0,
        created_at=now_iso(),
    )

    with jobs_lock:
        jobs[job_id] = job

    job_queue.put(job_id)

    return jsonify({"ok": True, "job": asdict(job)})


@app.get("/api/jobs")
def api_jobs():
    with jobs_lock:
        items = [asdict(j) for j in sorted(jobs.values(), key=lambda x: x.created_at, reverse=True)]
    return jsonify({"jobs": items})


@app.get("/api/files")
def api_files():
    return jsonify({"files": list_outputs()})


@app.get("/audio/<path:filename>")
def audio(filename: str):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(debug=True, port=58317)
