"""Qwen3-VL validation ablation for Challenge 2702."""
from __future__ import annotations
import json,re,tempfile,subprocess
from pathlib import Path
import modal
from videoqa_mc.mcqa_core import make_answer
ROOT=Path('/data/2702-multiple-choice-videoqa')
MODEL='Qwen/Qwen3-VL-8B-Instruct'
app=modal.App('unified-mcqa-2702-qwen3')
volume=modal.Volume.from_name('perception-test-2026-data',create_if_missing=False)
image=(modal.Image.debian_slim(python_version='3.11').apt_install('ffmpeg').pip_install('accelerate>=1.9','qwen-vl-utils','torch==2.7.1','torchvision==0.22.1','transformers>=4.57.1').env({'HF_HOME':str(ROOT/'models/huggingface')}).add_local_python_source('videoqa_mc'))

def _video(key,source):
 for root in (ROOT/'raw/validation',ROOT/'raw/full_validation'):
  for ident in (key,source):
   h=list(root.rglob(f'{ident}*.mp4'))
   if h:return h[0]
 raise FileNotFoundError(key)
def _prompt(q):
 opts='\n'.join(f'{chr(65+i)}. {x}' for i,x in enumerate(q['options']))
 return f'Inspect the entire video carefully. Answer with exactly one option letter.\nQuestion: {q["question"]}\nOptions:\n{opts}'
def _letter(s,n):
 m=re.search(r'\b([A-E])\b',s.upper())
 if not m: raise ValueError(s)
 i=ord(m.group(1))-65
 if i>=n: raise ValueError(s)
 return i

def _frames(video,n=24):
 import subprocess
 out=[]; d=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(video)],text=True).strip())
 for i in range(n):
  p=Path('/tmp')/f'q3_{i}.jpg'; t=d*i/max(1,n-1); subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-ss',f'{min(t,max(0,d-.05)):.3f}','-i',str(video),'-frames:v','1','-vf','scale=768:-2','-q:v','3',str(p)],check=True); out.append(p)
 return out
@app.function(image=image,volumes={'/data':volume})
def ping():
 import os
 return {'ok': True, 'data': os.listdir('/data')[:20]}

@app.function(image=image,gpu='A10G',volumes={'/data':volume},timeout=24*60*60)
def answer_validation(limit:int=0,record_start:int=0,record_end:int=0,prefix:str='qwen3',frame_count:int=12):
 import torch
 from PIL import Image
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 volume.reload(); db=json.loads(next((ROOT/'manifests/validation_annotations').glob('*.json')).read_text())
 model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,torch_dtype='auto',device_map='auto').eval(); processor=AutoProcessor.from_pretrained(MODEL)
 done=0; records=list(db.items())[record_start:record_end or None]
 for key,r in records:
  try:
   video=_video(key,str(r.get('metadata',{}).get('video_id',''))); paths=_frames(video,frame_count)
  except (FileNotFoundError, subprocess.CalledProcessError):
   print(f'SKIP unavailable video: {key}',flush=True)
   continue
  images=[Image.open(p).convert('RGB') for p in paths]
  for i,q in enumerate(r.get('mc_question',[])):
   dest=ROOT/'predictions'/'validation'/f'{prefix}_{key}_{i}.json'
   if dest.exists(): done+=1; continue
   messages=[{'role':'user','content':[{'type':'image','image':im} for im in images]+[{'type':'text','text':_prompt(q)}]}]
   text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
   inputs=processor(text=[text],images=images,padding=True,return_tensors='pt').to(model.device)
   with torch.inference_mode():
    gen=model.generate(**inputs,max_new_tokens=8,return_dict_in_generate=True,output_scores=True)
   raw=processor.batch_decode(gen.sequences[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0].strip(); aid=_letter(raw,len(q['options']))
   # First-step A-E logits provide a label-free confidence signal for routing.
   letter_ids=[]
   for letter in 'ABCDE'[:len(q['options'])]:
    ids=processor.tokenizer(letter,add_special_tokens=False).input_ids
    letter_ids.append(ids[-1])
   logits=gen.scores[0][0,letter_ids]; probs=torch.softmax(logits,dim=0).detach().cpu().tolist()
   confidence={'choice_probs':probs,'top1':max(probs),'margin':sorted(probs,reverse=True)[0]-sorted(probs,reverse=True)[1]}
   dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps({'record_key':key,'question_index':i,'answer':make_answer(q,aid),'model':MODEL,'raw_output':raw,'confidence':confidence}))
   done+=1
   if limit and done>=limit: break
  if limit and done>=limit: break
  if done%8==0: volume.commit()
 volume.commit(); return {'processed':done,'model':MODEL}
