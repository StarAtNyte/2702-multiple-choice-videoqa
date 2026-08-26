"""Gemini-via-Vertex MCQA scorer for Challenge 2702.

Structured-output enum (not free-text parsing) avoids the letter-parsing bug
class hit earlier this session with Qwen3-32B: the model can only return one
of the exact option letters, so there is nothing to regex. Layers two
techniques pulled from research this session (see RESEARCH.md): option-
permutation TTA + majority vote (first-place Perception Test MC-VQA
solution, arXiv:2409.13538) and a fallback-to-second-model pattern (matches
the organizer-reported best zero-shot technique, arXiv:2601.06287) using
already-cached Molmo predictions when Gemini fails after retries.

Implement-only: not run yet. Needs MODAL_PROFILE=baptinjean (the Modal
workspace holding the gcp-vertex-service-account secret; see
../gemini-vertex-api-setup.md). Vertex-only, no AI-Studio key path.
"""
from __future__ import annotations
import hashlib, json, os, random, sys, tempfile, traceback
import time
from pathlib import Path
from typing import Any
import modal
from mcqa_core import make_answer
from question_planner import classify

ROOT = Path('/data/2702-multiple-choice-videoqa')
MODEL = 'gemini-3.7-flash'
VERTEX_PROJECT = os.environ.get('VERTEX_PROJECT', 'veotest-460812')
VERTEX_LOCATION = os.environ.get('VERTEX_LOCATION', 'global')
GCP_SECRET_NAME = os.environ.get('GCP_MODAL_SECRET', 'gcp-vertex-service-account')
app = modal.App('unified-mcqa-2702-gemini')
vol = modal.Volume.from_name('perception-test-2026-data', create_if_missing=False)
img = (modal.Image.debian_slim(python_version='3.11').apt_install('ffmpeg')
       .pip_install('google-genai', 'google-auth')
       .add_local_python_source('mcqa_core')
       .add_local_python_source('question_planner'))

# split -> (annotation glob-or-path, video roots to search, prediction subdir,
# molmo fallback prediction dir)
SPLITS = {
 'validation': (('manifests/validation_annotations', '*.json'), ('raw/full_validation', 'raw/validation'), 'predictions/validation', 'predictions/validation'),
 'test_original': (('manifests/test_annotations', 'mc_question_test.json'), ('raw/test',), 'predictions/test_gemini', 'predictions/test'),
 'test_additional': (('manifests', 'mc_vqa_unified_test.json'), ('raw/test_unified',), 'predictions/test_unified_gemini', 'predictions/test_unified'),
}

def annotations(split):
 sub, name = SPLITS[split][0]
 d = ROOT / sub
 return json.loads((d / name if name != '*.json' else next(d.glob('*.json'))).read_text())

def find_video(split, key, source):
 for root_name in SPLITS[split][1]:
  root = ROOT / root_name
  if not root.exists(): continue
  for ident in (key, source):
   for p in root.rglob(f'{ident}*.mp4'):
    if p.stat().st_size > 10000: return p
 raise FileNotFoundError(key)

def permute(options, seed):
 rng = random.Random(seed)
 order = list(range(len(options)))
 rng.shuffle(order)
 return order, [options[i] for i in order]

def inference_plan(q: dict[str, Any], default_fps: float, default_scale: int,
                   adaptive: bool) -> tuple[float, int, str]:
 """Return query-adaptive media settings and a prompt skill.

 The planner is label-free: it only reads the question text/tags.  Defaults
 remain available for controlled ablations and reproducibility.
 """
 route = classify(q)
 if not adaptive:
  return default_fps, default_scale, route.skill
 fps_by_skill = {
  'audio': 1.0, 'temporal': 3.0, 'point_tracking': 4.0,
  'counting': 2.0, 'interaction': 2.0, 'general': 1.0,
 }
 scale_by_skill = {'point_tracking': 640, 'interaction': 576, 'temporal': 512}
 return fps_by_skill.get(route.skill, default_fps), scale_by_skill.get(route.skill, default_scale), route.skill

def clip_bytes(source, scale=480):
 # Downscale to bound the Vertex inline-request payload; preserve the source
 # audio because Challenge 2702 explicitly includes sound/speech questions.
 # FPS sampling is
 # handled by video_metadata.fps at request time, not by pre-subsampling
 # frames here (native-density source lets Gemini pick its own temporal
 # tokens, same reasoning as the CoTracker3 native-density fix earlier).
 import subprocess
 tmp = Path(tempfile.mktemp(suffix='.mp4'))
 subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(source),
  '-vf', f'scale={scale}:-2', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
  '-c:a', 'aac', '-b:a', '64k',
  '-pix_fmt', 'yuv420p', '-y', str(tmp)], check=True, timeout=180)
 data = tmp.read_bytes(); tmp.unlink(missing_ok=True)
 return data

def gemini_client():
 from google import genai
 from google.genai import types
 creds_json = os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON']
 creds_path = Path(tempfile.gettempdir()) / 'gcp-vertex-sa.json'
 creds_path.write_text(creds_json)
 os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(creds_path)
 return genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION,
  http_options=types.HttpOptions(timeout=180_000))

TRANSIENT_MARKERS = (
 '429', 'quota', 'rate limit', 'ratelimit', 'resource exhausted',
 'too many requests', 'temporarily unavailable', 'deadline exceeded',
 'timeout', 'timed out', '500', '502', '503', '504',
)

def is_transient(exc: Exception) -> bool:
 text = str(exc).lower()
 return any(marker in text for marker in TRANSIENT_MARKERS)

def ask(client, video_data, fps, scale, q, order, model, thinking_level, skill='general',
        max_attempts: int = 2, backoff_base: float = 10.0, backoff_cap: float = 10.0):
 from google.genai import types
 letters = 'ABCDE'[:len(q['options'])]
 opts_text = '\n'.join(f'{letters[i]}. {opt}' for i, opt in enumerate([q['options'][j] for j in order]))
 focus = {
  'audio': 'Use both the audio and visual streams; distinguish speech from sound events.',
  'temporal': 'Build a brief timestamped event order internally before choosing.',
  'point_tracking': 'Inspect early and late frames; follow the highlighted surface point.',
  'counting': 'Count only visible instances and cross-check across frames.',
  'interaction': 'Identify the objects visibly participating in the highlighted action.',
 }.get(skill, 'Inspect the entire video carefully, including temporal changes.')
 prompt = (focus + ' Reason silently, compare every option against the evidence, '
  'and return only the single option letter as the final response.\nQuestion: '
  + q['question'] + '\nOptions:\n' + opts_text)
 parts = [types.Part(inline_data=types.Blob(data=video_data, mime_type='video/mp4'),
  video_metadata=types.VideoMetadata(fps=fps)), types.Part.from_text(text=prompt)]
 config = types.GenerateContentConfig(
  thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
  temperature=0.0,
  media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW if scale < 512 else types.MediaResolution.MEDIA_RESOLUTION_HIGH,
  response_mime_type='text/x.enum', response_schema={'type': 'STRING', 'enum': list(letters)},
  max_output_tokens=1024)
 response = None
 for attempt in range(max_attempts):
  try:
   response = client.models.generate_content(model=model, contents=[types.Content(role='user', parts=parts)], config=config)
   break
  except Exception as exc:
   if not is_transient(exc) or attempt == max_attempts - 1:
    raise
   # Jitter prevents a whole Modal batch from retrying in lockstep after a
   # shared quota window reopens. No output is written during this loop.
   delay = min(backoff_cap, backoff_base * (2 ** attempt))
   delay *= random.uniform(0.8, 1.2)
   print(f'[retry] transient Gemini error attempt={attempt + 1}/{max_attempts} '
         f'sleep={delay:.1f}s error={exc!r}', flush=True)
   time.sleep(delay)
 letter = (response.text or '').strip().upper()
 permuted_index = letters.index(letter)  # raises ValueError if malformed; caller catches
 return order[permuted_index]

def molmo_fallback(fallback_dir, key, i):
 base = ROOT / fallback_dir
 # Historical validation runs used several explicit prefixes; prefer the
 # canonical file, then the cached low-resolution Molmo run.
 for prefix in ('molmo_', 'molmo_low_', 'molmo_specialist_'):
  path = base / f'{prefix}{key}_{i}.json'
  if path.exists():
   return json.loads(path.read_text())['answer']
 return None

@app.function(image=img, secrets=[modal.Secret.from_name(GCP_SECRET_NAME)], volumes={'/data': vol}, timeout=20 * 3600)
def run_split(split: str = 'validation', prefix: str = 'gemini', model: str = MODEL,
 tta_rounds: int = 1, thinking_level: str = 'low', fps: float = 0.5, scale: int = 480,
 limit: int = 0, adaptive: bool = False, fallback_first: bool = False,
 shard_id: int = -1, num_shards: int = 1, retry_only: bool = False):
 vol.reload()
 client = gemini_client()
 out_dir = ROOT / SPLITS[split][2]
 out_dir.mkdir(parents=True, exist_ok=True)
 db = annotations(split)
 done = failed = fallback_used = 0
 if num_shards < 1 or shard_id >= num_shards:
  raise ValueError(f'invalid shard_id={shard_id} num_shards={num_shards}')
 records = list(db.items())
 if shard_id >= 0:
  records = [(key, record) for index, (key, record) in enumerate(records)
             if index % num_shards == shard_id]
 print(f'[start] split={split} shard={shard_id}/{num_shards} records={len(records)}', flush=True)
 for key, r in records:
  try:
   video = find_video(split, key, str(r.get('metadata', {}).get('video_id', '')))
  except FileNotFoundError:
   print(f'[skip] video not found key={key}', flush=True); continue
  video_data = None
  for i, q in enumerate(r.get('mc_question', [])):
   if limit and done >= limit:
    print(f'[done] processed={done} failed={failed} fallback={fallback_used}', flush=True)
    return {'processed': done, 'failed': failed, 'fallback': fallback_used}
   out = out_dir / f'{prefix}_{key}_{i}.json'
   err_out = out_dir / f'{prefix}_{key}_{i}.error.json'
   # Successful question outputs are resumable checkpoints. Error records are
   # deliberately retried on the next invocation so a quota window can heal.
   if out.exists() or (retry_only and not err_out.exists()): continue
   try:
    qfps, qscale, skill = inference_plan(q, fps, scale, adaptive)
    if fallback_first:
     fallback = molmo_fallback(SPLITS[split][3], key, i)
     if fallback is not None:
      out.parent.mkdir(parents=True, exist_ok=True)
      out.write_text(json.dumps({'record_key': key, 'question_index': i, 'answer': fallback,
       'model': 'molmo_fallback_first', 'adaptive': adaptive, 'skill': skill}, sort_keys=True))
      done += 1
      print(f'[fallback-first] key={key} q={i} done={done} failed={failed}', flush=True)
      continue
    # Encode once per source video at the highest requested scale. Gemini's
    # video_metadata.fps controls temporal sampling at request time.
    if video_data is None: video_data = clip_bytes(video, scale=max(scale, qscale))
    votes: dict[int, int] = {}
    raw_votes = []
    for round_i in range(max(1, tta_rounds)):
     seed = int(hashlib.sha1(f'{key}:{i}:{round_i}'.encode()).hexdigest(), 16) % (2 ** 31)
     order, _ = permute(q['options'], seed)
     aid = ask(client, video_data, qfps, qscale, q, order, model, thinking_level, skill)
     votes[aid] = votes.get(aid, 0) + 1
     raw_votes.append(aid)
    aid = max(votes, key=lambda k: (votes[k], -raw_votes.index(k)))
   except Exception as exc:
    fallback = molmo_fallback(SPLITS[split][3], key, i)
    if fallback is not None:
     fallback_used += 1
     out.parent.mkdir(parents=True, exist_ok=True)
     out.write_text(json.dumps({'record_key': key, 'question_index': i, 'answer': fallback,
      'model': 'molmo_fallback', 'gemini_error': repr(exc)}))
     done += 1
     print(f'[fallback] key={key} q={i} done={done} failed={failed} fallback={fallback_used}', flush=True)
     continue
    failed += 1
    print(f'[error] key={key} q={i} exc={exc!r}', file=sys.stderr, flush=True)
    err_out.parent.mkdir(parents=True, exist_ok=True)
    err_out.write_text(json.dumps({'record_key': key, 'question_index': i, 'model': model,
     'error': repr(exc), 'traceback': traceback.format_exc()}))
    continue
   out.parent.mkdir(parents=True, exist_ok=True)
   err_out.unlink(missing_ok=True)
   out.write_text(json.dumps({'record_key': key, 'question_index': i, 'answer': make_answer(q, aid),
    'model': model, 'tta_rounds': tta_rounds, 'votes': {str(k): v for k, v in votes.items()},
    'adaptive': adaptive, 'skill': skill, 'fps': qfps, 'scale': qscale}, sort_keys=True))
   done += 1
   print(f'[ok] key={key} q={i} done={done} failed={failed} fallback={fallback_used}', flush=True)
  vol.commit()
 vol.commit()
 print(f'[complete] processed={done} failed={failed} fallback={fallback_used}', flush=True)
 return {'processed': done, 'failed': failed, 'fallback': fallback_used}

@app.function(image=img, volumes={'/data': vol}, timeout=1800)
def fill_missing_from_molmo(split: str = 'validation', prefix: str = 'gemini37_adaptive_tta3'):
 """Fill only missing output slots from already-cached Molmo predictions.

 This is a no-network recovery path for quota outages. It never overwrites a
 Gemini answer and never invents an answer when the cache has no exact key.
 """
 vol.reload()
 out_dir = ROOT / SPLITS[split][2]
 db = annotations(split)
 filled = missing = 0
 for key, record in db.items():
  for i, q in enumerate(record.get('mc_question', [])):
   out = out_dir / f'{prefix}_{key}_{i}.json'
   if out.exists():
    continue
   fallback = molmo_fallback(SPLITS[split][3], key, i)
   if fallback is None:
    missing += 1
    continue
   out.parent.mkdir(parents=True, exist_ok=True)
   out.write_text(json.dumps({'record_key': key, 'question_index': i,
    'answer': fallback, 'model': 'molmo_recovery'}, sort_keys=True))
   filled += 1
 vol.commit()
 return {'filled': filled, 'unresolved': missing}

@app.function(image=img, volumes={'/data': vol}, timeout=600)
def audit_checkpoints(split: str = 'test_original', prefix: str = 'gemini37_test_whitestjohn_adaptive'):
 vol.reload()
 db = annotations(split)
 expected = {(key, i) for key, record in db.items() for i, _ in enumerate(record.get('mc_question', []))}
 out_dir = ROOT / SPLITS[split][2]
 valid_success = valid_error = orphan = 0
 seen = set()
 for path in out_dir.glob(f'{prefix}_*.json'):
  try:
   payload = json.loads(path.read_text())
   pair = (payload.get('record_key'), int(payload.get('question_index')))
  except Exception:
   orphan += 1; continue
  if pair not in expected:
   orphan += 1; continue
  seen.add(pair)
  if path.name.endswith('.error.json'): valid_error += 1
  else: valid_success += 1
 return {'expected': len(expected), 'valid_success': valid_success, 'valid_error': valid_error,
         'missing': len(expected - seen), 'orphan_files': orphan}
