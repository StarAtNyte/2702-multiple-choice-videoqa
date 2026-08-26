"""Label-free deterministic specialists for the unified five-choice questions."""
from __future__ import annotations
import json, re, subprocess
from pathlib import Path
from typing import Any
import modal
from mcqa_core import make_answer

ROOT = Path('/data/2702-multiple-choice-videoqa')
app = modal.App('unified-mcqa-2702-specialists')
volume = modal.Volume.from_name('perception-test-2026-data', create_if_missing=False)
image = (modal.Image.debian_slim(python_version='3.11')
    .apt_install('ffmpeg')
    .pip_install('opencv-python-headless>=4.10', 'numpy<2')
    .add_local_python_source('mcqa_core'))

COLORS = {
    'red': (0, 10), 'orange': (8, 22), 'yellow': (22, 42), 'green': (38, 85),
    'cyan': (85, 105), 'blue': (100, 135), 'magenta': (135, 175), 'purple': (125, 165)
}

def _video(root: Path, key: str, source: str) -> Path:
    for ident in (key, source):
        hits = list(root.rglob(f'{ident}*.mp4'))
        if hits: return hits[0]
    raise FileNotFoundError(key)

def _frames(video: Path, n: int = 12):
    import cv2
    cap = cv2.VideoCapture(str(video)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if total <= 0: raise RuntimeError(video)
    out=[]
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * (total-1) / max(1,n-1)))
        ok, frame = cap.read()
        if ok: out.append(frame)
    cap.release(); return out

def _hue(name: str) -> tuple[int,int]:
    for k,v in COLORS.items():
        if k in name.lower(): return v
    return (0,180)

def _centers(frame, name: str):
    import cv2, numpy as np
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV); lo,hi=_hue(name)
    if lo <= hi: mask=cv2.inRange(hsv,(lo,80,80),(hi,255,255))
    else: mask=cv2.inRange(hsv,(lo,80,80),(179,255,255))|cv2.inRange(hsv,(0,80,80),(hi,255,255))
    mask=cv2.medianBlur(mask,5); contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    result=[]
    for c in contours:
        area=cv2.contourArea(c)
        if area < 8: continue
        (x,y),r=cv2.minEnclosingCircle(c)
        if r >= 2: result.append((float(x),float(y),float(area)))
    return result

def _initial_point(frames, color: str):
    candidates=[]
    for frame in frames[:max(2,len(frames)//4)]: candidates += _centers(frame,color)
    if not candidates: return None
    # Marker is usually the compact, saturated component; prefer median location of small blobs.
    candidates.sort(key=lambda x: x[2])
    return candidates[len(candidates)//2][:2]

def _track(frames, point):
    import cv2, numpy as np
    p=np.array([[point]],dtype=np.float32); prev=cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        nxt,st,err=cv2.calcOpticalFlowPyrLK(prev,gray,p,None,winSize=(31,31),maxLevel=3,
                                             criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.03))
        if nxt is None or not st[0,0]: break
        p=nxt; prev=gray
    return tuple(map(float,p[0,0]))

def _point_answer(question, frames):
    text=question['question']; m=re.search(r'\b(Green|Magenta|Red|Yellow|Blue|Cyan) circle',text,re.I)
    if not m: return None
    initial=_initial_point(frames,m.group(1))
    if initial is None: return None
    tracked=_track(frames,initial); final=frames[-1]
    options=question['options']; scored=[]
    for i,opt in enumerate(options):
        c=_centers(final,opt)
        if not c: scored.append((1e9,i)); continue
        scored.append((min((x-tracked[0])**2+(y-tracked[1])**2 for x,y,_ in c),i))
    return min(scored)[1]

@app.function(image=image, volumes={'/data':volume}, timeout=3600)
def validation_point(limit:int=0, frame_count:int=12, prefix:str='specialist_point')->dict[str,Any]:
    volume.reload(); ann=json.loads(next((ROOT/'manifests/validation_annotations').glob('*.json')).read_text())
    out=ROOT/'predictions'/'validation'; done=correct=total=0
    for key,r in ann.items():
        video=None
        for vr in (ROOT/'raw/validation', ROOT/'raw/full_validation'):
            try: video=_video(vr,key,str(r.get('metadata',{}).get('video_id',''))); break
            except FileNotFoundError: pass
        if video is None: raise FileNotFoundError(key)
        for i,q in enumerate(r.get('mc_question',[])):
            if len(q.get('options',[]))!=5 or 'point_tracking' not in ' '.join(q.get('tag',[])): continue
            frames=_frames(video, frame_count); aid=_point_answer(q,frames)
            if aid is None: continue
            total+=1; correct+=int(aid==q['answer_id']); done+=1
            (out/f'{prefix}_{key}_{i}.json').write_text(json.dumps({'record_key':key,'question_index':i,'answer':make_answer(q,aid),'method':'opencv-optical-flow'}))
            if limit and done>=limit: break
        if limit and done>=limit: break
    volume.commit(); return {'processed':done,'scored':total,'correct':correct,'accuracy':correct/total if total else 0}

@app.function(image=image, volumes={'/data':volume}, timeout=300)
def evaluate_point() -> dict[str, Any]:
    volume.reload(); ann=json.loads(next((ROOT/'manifests/validation_annotations').glob('*.json')).read_text())
    correct=total=0
    for path in (ROOT/'predictions'/'validation').glob('specialist_point_*.json'):
        r=json.loads(path.read_text()); q=ann[r['record_key']]['mc_question'][r['question_index']]
        total += 1; correct += int(r['answer']['answer_id'] == q['answer_id'])
    return {'correct':correct,'total':total,'accuracy':correct/total if total else 0}

@app.local_entrypoint()
def main(action: str = 'point'):
    print(evaluate_point.remote() if action == 'evaluate' else validation_point.remote())
