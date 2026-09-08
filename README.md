# Multimodal Sentiment Analysis

Reads sentiment from four sources — written text, facial expression, tone of
voice and video — and fuses them into a single verdict.

Every modality returns the same structured result: a `positive / neutral /
negative` distribution, a `-1 … +1` valence score, a confidence, and the
evidence behind it.

---

## Quick start

```bash
python -m venv venv
```

Activate it, then:

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

Open <http://127.0.0.1:5000>.

**Windows note.** Set `PYTHONIOENCODING=utf-8` before running. DeepFace logs
emoji, the default `cp1252` console cannot encode them, and the resulting
`UnicodeEncodeError` is reported misleadingly as a model *download* failure.

```powershell
$env:PYTHONIOENCODING="utf-8"
python app.py
```

PowerShell 5.1 has no `&&`; chain commands with `;` instead.

### Audio decoding

No system ffmpeg install is required: `imageio-ffmpeg` ships a static binary,
and the app adds it to `PATH` on first use. The home page badge shows whether a
decoder was found.

Note that **file extensions lie**. WhatsApp exports MPEG-4/AAC audio named
`.mp3`; libsndfile rejects it because it is not MPEG audio, and `audioread`
then raises an error with an empty message. The app sniffs the magic bytes and
reports the real container instead of a blank error.

---

## How the analysis works

### Shared scoring

Each model emits its own emotion labels. Every label is placed on a valence
axis from −1 to +1, and its distance from zero decides how much it counts as
*neutral*:

| contribution | formula |
| --- | --- |
| positive | `p × max(v, 0)` |
| negative | `p × max(−v, 0)` |
| neutral  | `p × (1 − abs(v))` |

So `happy` (v = 1.0) is fully positive, `angry` (−0.9) is almost fully
negative, and `surprise` (0.2) stays mostly neutral instead of being forced
into a polarity it does not carry. The result is always a valid probability
distribution.

A winning label whose valence falls inside `NEUTRAL_BAND` is reported as
neutral: "very slightly positive" is not a useful verdict.

Alongside confidence, each result carries a **certainty** derived from the
distribution's entropy — 1.0 when the model is decisive, 0.0 when it is
guessing between all three.

### Text

- 3-class model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) so
  factual statements can be *neutral*. A binary SST-2 model has no neutral
  class and labels "The package arrived on Tuesday" as positive with high
  confidence. If it cannot be downloaded the app falls back to SST-2 and says
  so in the result warnings.
- Input longer than the model's 512-token limit is split into **overlapping
  windows** and recombined, weighted by window length, rather than truncated.

### Audio

- Resampled to 16 kHz mono, then cut into overlapping windows, because the
  classifier was trained on short utterances — feeding it a three-minute file
  yields one meaningless average.
- Windows are weighted by **loudness**, and near-silent ones are dropped: quiet
  passages carry no emotional evidence, yet the classifier still labels them.
- With `ENABLE_ASR=1` the speech is transcribed and *what was said* is fused
  with *how it was said*.

### Image

- Every detected face is scored, not just the first.
- Faces are weighted by area × detection confidence, so the subject of the
  photo outweighs a bystander in the background.
- If nothing is detected, the request **fails with an explanation** rather than
  returning a verdict. DeepFace's lenient mode hands back the entire frame with
  `face_confidence` of 0 when it finds no face; running the emotion model on an
  uncropped photo yields a confident but meaningless answer (a smiling cartoon
  scored "angry 97.7%"), so zero-confidence records are discarded.
- Detectors are trained on **photographs**. Drawings, cartoons, avatars and
  heavily stylised faces are usually not detected at all — this is a property
  of the detector, not a bug in the app.

### Video

- Frames are sampled evenly across the whole clip under a hard budget
  (`VIDEO_MAX_FRAMES`), so a two-hour film is *summarised* rather than
  truncated at the first N frames, and the runtime stays bounded.
- Per-frame emotions form a timeline; the soundtrack is analysed separately and
  fused with the visual verdict when ffmpeg is present.
- Runs as a **background job** with progress polling — a long clip would
  otherwise exceed any HTTP timeout.

### Fusion

`fuse()` combines modality results weighted by configured importance *and* by
each modality's certainty, so a decisive signal outweighs a hesitant one. It
reports whether the modalities agreed and warns when they did not.

---

## API

| Method | Route | Purpose |
| --- | --- | --- |
| `GET`  | `/` | Web interface |
| `GET`  | `/api/health` | Status, ffmpeg availability, loaded models |
| `POST` | `/api/analyze/text` | `{"text": "..."}` → result |
| `POST` | `/api/analyze/image` | multipart `file` → result |
| `POST` | `/api/analyze/audio` | multipart `file` → result |
| `POST` | `/api/analyze/video` | multipart `file` → `{"job_id": "..."}` (202) |
| `GET`  | `/api/jobs/<id>` | Job progress and result |
| `POST` | `/api/fuse` | `{"results": {...}}` → combined verdict |

```bash
curl -X POST http://127.0.0.1:5000/api/analyze/text -H "Content-Type: application/json" -d "{\"text\":\"I love this\"}"
```

Result shape:

```json
{
  "ok": true,
  "modality": "text",
  "label": "positive",
  "confidence": 0.981,
  "valence": 0.961,
  "certainty": 0.87,
  "distribution": {"positive": 0.981, "neutral": 0.017, "negative": 0.002},
  "detail": {"words": 3, "windows": 1},
  "warnings": [],
  "error": null,
  "elapsed_ms": 42
}
```

Failures use the same shape with `"ok": false` and a human-readable `error`,
so clients never need two code paths.

---

## Configuration

Every value in `config.py` can be set by an environment variable of the same
name; see `.env.example`. The ones worth knowing:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAX_UPLOAD_MB` | 100 | Rejected before the file is written to disk |
| `VIDEO_MAX_FRAMES` | 120 | Frame budget per clip |
| `VIDEO_MAX_SECONDS` | 900 | Ignore anything past this point |
| `AUDIO_MAX_SECONDS` | 600 | Same, for audio |
| `NEUTRAL_BAND` | 0.15 | Valence below this reports as neutral |
| `ENABLE_ASR` | 0 | Transcribe speech and fuse it (extra download) |
| `FACE_DETECTOR` | opencv | `retinaface` and `mtcnn` are slower but better |
| `WEIGHT_TEXT/AUDIO/FACE` | 1.0/0.8/0.9 | Fusion weights |

---

## Running it elsewhere

The app is a normal Flask service, so any host that can run a container works.
There is a `Dockerfile`, a `Procfile` and `gunicorn.conf.py` in the repo.

```bash
docker build -t sentiment .
```

```bash
docker run -p 7860:7860 sentiment
```

### What will not work

| Host | Why |
| --- | --- |
| Vercel, Netlify | 500 MB function limit; this stack is ~7.9 GB |
| Render free tier | 512 MB RAM; the models alone need ~1.5 GB |
| Hugging Face Spaces | Creating Docker/Gradio Spaces now requires PRO ($9/mo). ZeroGPU, the remaining free tier, only runs Gradio |

Hosts that do work: Google Cloud Run (free tier, scales to zero, ~30-90 s cold
start), Render or Railway on a paid instance, Fly.io, or any VPS. Give it at
least **2 GB RAM**.

### Production notes

- **Set `SECRET_KEY`.** The default is a development placeholder.
- `gunicorn.conf.py` uses threads rather than many processes: each worker loads
  its own copy of the models, so workers are expensive in RAM. Budget roughly
  1.5-2 GB per worker.
- The request timeout is 1800s because video analysis is genuinely slow. Put a
  reverse proxy in front with a matching timeout.
- Jobs are held **in memory**, so they do not survive a restart and do not work
  across multiple machines. For multi-instance deployments, move the job
  registry in `app.py` to Redis or a database.
- Uploads are deleted as soon as analysis finishes.

## Tests

```bash
python -m pytest tests/ -q
```

The default run is model-free and takes under a second. To include the
model-backed tests:

```bash
RUN_SLOW=1 python -m pytest tests/ -q
```

---

## Limitations

Worth being honest about:

- Facial emotion recognition is trained largely on posed, frontal,
  well-lit Western faces. Accuracy drops on candid images, profiles, poor
  lighting, occlusion, and is known to vary across demographic groups.
- The vocal model classifies four states (happy, neutral, sad, angry) and is
  English-centric; tone is culturally specific.
- Text sentiment reads surface polarity. Sarcasm, irony and negation over long
  distances are still failure cases.
- A confident-looking score is not a correct one. Treat output as a signal to
  review, not a verdict — and do not use it for consequential decisions about
  individuals.
