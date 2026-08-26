"""CoTracker3 learned point-tracking specialist for Challenge 2702."""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any
import modal

ROOT=Path('/data/2702-multiple-choice-videoqa')
app=modal.App('unified-mcqa-cotracker')
volume=modal.Volume.from_name('perception-test-2026-data',create_if_missing=False)
image=(modal.Image.debian_slim(python_version='3.11').apt_install('ffmpeg').pip_install('torch','torchvision','opencv-python-headless>=4.10','numpy<2'))
COLORS={'red':(0,10),'orange':(8,22),'yellow':(22,42),'green':(38,85),'cyan':(85,105),'blue':(100,135),'magenta':(135,175),'purple':(125,165)}
def hue(name): return COLORS.get(next((k for k in COLORS if k in name.lower()),'red'),(0,10))
def centers(frame,name):
 import cv2,numpy as np
 lo,hi=hue(name); hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
 mask=cv2.inRange(hsv,(lo,70,70),(hi,255,255)) if lo<=hi else cv2.inRange(hsv,(lo,70,70),(179,255,255))|cv2.inRange(hsv,(0,70,70),(hi,255,255))
 cs,_=cv2.findContours(cv2.medianBlur(mask,5),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); out=[]
 for c in cs:
  a=cv2.contourArea(c)
  if a>=8:
   (x,y),r=cv2.minEnclosingCircle(c)
   if r>=2: out.append((float(x),float(y),float(a)))
 return out
def video_path(key,source):
 for root in (ROOT/'raw/full_validation',ROOT/'raw/validation'):
  for ident in (key,source):
   hit=next(root.rglob(f'{ident}*.mp4'),None)
   if hit:return hit
 raise FileNotFoundError(key)
def load_frames(path,max_frames=2000):
 # CoTracker3 tracks by frame-to-frame motion continuity; the previous 64-way
 # uniform resample over a 35s clip left multi-frame gaps the tracker could
 # not bridge, so it lost the marker (scored below the 20% chance floor).
 # Read sequentially at native density and only stride down if the clip is
 # longer than max_frames, keeping gaps small either way.
 import cv2,numpy as np
 cap=cv2.VideoCapture(str(path)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 stride=max(1,total//max_frames) if total>0 else 1
 frames=[]; i=0
 while True:
  ok,f=cap.read()
  if not ok: break
  if i%stride==0: frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
  i+=1
 cap.release(); return frames,total
@app.function(image=image,gpu='H200',volumes={'/data':volume},timeout=7200)
def validation_point_cotracker(limit:int=0)->dict[str,Any]:
 import sys,traceback
 import torch,numpy as np
 model=torch.hub.load('facebookresearch/co-tracker','cotracker3_offline',pretrained=True).to('cuda').eval()
 volume.reload(); ann=json.loads(next((ROOT/'manifests/validation_annotations').glob('*.json')).read_text()); out=ROOT/'predictions/validation'; done=correct=failed=0
 for key,r in ann.items():
  try:
   path=video_path(key,str(r.get('metadata',{}).get('video_id','')))
  except FileNotFoundError:
   print(f'[skip] video not found key={key}',flush=True);continue
  for qi,q in enumerate(r.get('mc_question',[])):
   if len(q.get('options',[]))!=5 or 'point_tracking' not in ' '.join(q.get('tag',[])): continue
   m=re.search(r'\b(Green|Magenta|Red|Yellow|Blue|Cyan) circle',q['question'],re.I)
   if not m: continue
   try:
    frames,total=load_frames(path); import cv2
    cs=[]
    for f in frames[:max(2,len(frames)//4)]: cs+=centers(f,m.group(1))
    if not cs: raise ValueError('no initial marker detected')
    cs.sort(key=lambda x:x[2]); x,y=cs[len(cs)//2][:2]; h,w=frames[0].shape[:2]
    video=torch.from_numpy(np.stack(frames)).permute(0,3,1,2)[None].float().to('cuda')
    # query uses sampled-frame index and RGB pixel coordinates
    queries=torch.tensor([[[0.,x,y]]],device='cuda')
    with torch.inference_mode(): tracks,vis=model(video,queries=queries)
    tx,ty=tracks[0,-1,0].detach().cpu().tolist(); final=frames[-1]; scored=[]
    for oi,opt in enumerate(q['options']): scored.append((min(((cx-tx)**2+(cy-ty)**2 for cx,cy,_ in centers(final,opt)),default=1e12),oi))
    aid=min(scored)[1]
   except Exception as exc:
    failed+=1
    print(f'[error] key={key} q={qi} exc={exc!r}',file=sys.stderr,flush=True)
    (out/f'cotracker_point_{key}_{qi}.error.json').write_text(json.dumps({'record_key':key,'question_index':qi,
     'error':repr(exc),'traceback':traceback.format_exc()}))
    continue
   done+=1; correct+=int(aid==q.get('answer_id'))
   (out/f'cotracker_point_{key}_{qi}.json').write_text(json.dumps({'record_key':key,'question_index':qi,'answer_id':aid,'correct':aid==q.get('answer_id'),'method':'cotracker3_offline'}))
   print(f'[ok] key={key} q={qi} done={done} correct={correct} failed={failed}',flush=True)
   if limit and done>=limit: break
  if limit and done>=limit: break
 volume.commit()
 print(f'[complete] processed={done} correct={correct} failed={failed}',flush=True)
 return {'processed':done,'correct':correct,'failed':failed,'accuracy':correct/done if done else 0}
