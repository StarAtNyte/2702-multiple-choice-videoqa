"""Qwen3-VL-32B-Instruct-FP8 vLLM smoke/full validation."""
from __future__ import annotations
import json,re,sys,traceback
from pathlib import Path
import modal
from videoqa_mc.mcqa_core import make_answer
ROOT=Path('/data/2702-multiple-choice-videoqa'); MODEL='Qwen/Qwen3-VL-32B-Instruct-FP8'
app=modal.App('unified-mcqa-qwen3-32b-vllm')
vol=modal.Volume.from_name('perception-test-2026-data',create_if_missing=False)
img=(modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04', add_python='3.11').apt_install('ffmpeg').pip_install('vllm>=0.10.0','qwen-vl-utils','torchvision').env({'HF_HOME':str(ROOT/'models/huggingface'),'VLLM_USE_V1':'0'}).add_local_python_source('videoqa_mc'))
@app.function(image=modal.Image.debian_slim(python_version='3.11').add_local_python_source('videoqa_mc'),volumes={'/data':vol},timeout=300)
def purge(prefix:str='qwen3_32b_fp8_fps05'):
 vol.reload(); n=0
 for p in (ROOT/'predictions/validation').glob(f'{prefix}_*'):
  p.unlink(); n+=1
 vol.commit(); return {'deleted':n}
def find_video(key,source):
 for root in (ROOT/'raw/full_validation',ROOT/'raw/validation'):
  for ident in (key,source):
   for p in root.rglob(f'{ident}*.mp4'):
    if p.stat().st_size>10000:return p
 raise FileNotFoundError(key)
def prompt(q):
 return ('Inspect the entire video carefully, including temporal changes. Think step by step, '
  'then finish your reply on its own final line as: Answer: <letter>\nQuestion: '+q['question']
  +'\nOptions:\n'+'\n'.join(f'{chr(65+i)}. {x}' for i,x in enumerate(q['options'])))
def letter(s,n):
 # No unanchored fallback: scanning the whole CoT text for any standalone
 # A-E letter matches the English article "a" (upper-cased to "A") in
 # almost every sentence, which silently resolved 355/356 unanchored
 # answers to option A regardless of what the model actually concluded
 # (found via a post-run audit: fallback-bucket accuracy was 30.6% vs
 # 76.3% for anchored answers, and 99.7% of fallback answers were index 0).
 # Require the explicit anchor; anything without it is a genuine failure
 # that should retry with a bigger token budget, not a guess.
 m=re.search(r'ANSWER:\s*([A-E])\b',s.upper())
 if not m or ord(m.group(1))-65>=n: raise ValueError(s)
 return ord(m.group(1))-65
@app.function(image=img,gpu='H100',volumes={'/data':vol},timeout=12*3600)
def answer_validation(limit:int=8,prefix:str='qwen3_32b_fp8'):
 import array
 class GenericArray(array.array):
  @classmethod
  def __class_getitem__(cls,item): return cls
 array.array=GenericArray
 from vllm import LLM,SamplingParams
 from qwen_vl_utils import process_vision_info
 llm=LLM(model=MODEL,trust_remote_code=True,gpu_memory_utilization=.92,max_model_len=4096,max_num_seqs=1,enforce_eager=True)
 sampling=SamplingParams(temperature=0,max_tokens=400)
 vol.reload();db=json.loads(next((ROOT/'manifests/validation_annotations').glob('*.json')).read_text())
 done=failed=0
 for key,r in db.items():
  try:
   v=find_video(key,str(r.get('metadata',{}).get('video_id','')))
  except FileNotFoundError:
   print(f'[skip] video not found key={key}',flush=True);continue
  for i,q in enumerate(r.get('mc_question',[])):
   if limit and done>=limit:
    print(f'[done] processed={done} failed={failed} model={MODEL}',flush=True)
    return {'processed':done,'failed':failed,'model':MODEL}
   out=ROOT/'predictions/validation'/f'{prefix}_{key}_{i}.json'
   err_out=ROOT/'predictions/validation'/f'{prefix}_{key}_{i}.error.json'
   if out.exists() or err_out.exists():continue
   raw=None
   try:
    messages=[{'role':'user','content':[{'type':'video','video':str(v),'fps':0.5,'max_pixels':512*512},{'type':'text','text':prompt(q)}]}]
    text=llm.get_tokenizer().apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    _,videos,video_kwargs=process_vision_info(messages,return_video_kwargs=True,return_video_metadata=True)
    mm={'video':videos[0]}
    inp={'prompt':text,'multi_modal_data':mm}
    result=llm.generate([inp],sampling, use_tqdm=False)[0]
    raw=result.outputs[0].text.strip(); aid=letter(raw,len(q['options']))
   except Exception as exc:
    failed+=1
    print(f'[error] key={key} q={i} raw={raw!r} exc={exc!r}',file=sys.stderr,flush=True)
    err_out.parent.mkdir(parents=True,exist_ok=True)
    err_out.write_text(json.dumps({'record_key':key,'question_index':i,'model':MODEL,'raw_output':raw,
     'error':repr(exc),'traceback':traceback.format_exc()}))
    continue
   out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'record_key':key,'question_index':i,'answer':make_answer(q,aid),'model':MODEL,'raw_output':raw,'mode':'vllm_fp8'}));done+=1
   print(f'[ok] key={key} q={i} done={done} failed={failed}',flush=True)
  vol.commit()
 vol.commit()
 print(f'[complete] processed={done} failed={failed} model={MODEL}',flush=True)
 return {'processed':done,'failed':failed,'model':MODEL}
@app.function(image=img,gpu='H100',volumes={'/data':vol},timeout=4*3600)
def retry_failed(prefix:str='qwen3_32b_fp8_fps05',max_tokens:int=400):
 # Errors were mostly step-by-step CoT hitting the 128-token cap before the
 # Answer: <letter> anchor (counting/spelling/order questions need more room).
 # Re-run only the accumulated error files with a bigger budget instead of
 # redoing the full 454.
 import array
 class GenericArray(array.array):
  @classmethod
  def __class_getitem__(cls,item): return cls
 array.array=GenericArray
 from vllm import LLM,SamplingParams
 from qwen_vl_utils import process_vision_info
 llm=LLM(model=MODEL,trust_remote_code=True,gpu_memory_utilization=.92,max_model_len=4096,max_num_seqs=1,enforce_eager=True)
 sampling=SamplingParams(temperature=0,max_tokens=max_tokens)
 vol.reload();db=json.loads(next((ROOT/'manifests/validation_annotations').glob('*.json')).read_text())
 fixed=still_failed=0
 err_files=sorted((ROOT/'predictions/validation').glob(f'{prefix}_*.error.json'))
 print(f'[retry] {len(err_files)} error files to retry',flush=True)
 for err_out in err_files:
  rec=json.loads(err_out.read_text()); key=rec['record_key']; i=rec['question_index']
  q=db[key]['mc_question'][i]
  out=ROOT/'predictions/validation'/f'{prefix}_{key}_{i}.json'
  raw=None
  try:
   v=find_video(key,str(db[key].get('metadata',{}).get('video_id','')))
   messages=[{'role':'user','content':[{'type':'video','video':str(v),'fps':0.5,'max_pixels':512*512},{'type':'text','text':prompt(q)}]}]
   text=llm.get_tokenizer().apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
   _,videos,video_kwargs=process_vision_info(messages,return_video_kwargs=True,return_video_metadata=True)
   mm={'video':videos[0]}
   inp={'prompt':text,'multi_modal_data':mm}
   result=llm.generate([inp],sampling, use_tqdm=False)[0]
   raw=result.outputs[0].text.strip(); aid=letter(raw,len(q['options']))
  except Exception as exc:
   still_failed+=1
   print(f'[still-failed] key={key} q={i} raw={raw!r} exc={exc!r}',file=sys.stderr,flush=True)
   err_out.write_text(json.dumps({'record_key':key,'question_index':i,'model':MODEL,'raw_output':raw,
    'error':repr(exc),'traceback':traceback.format_exc(),'retries':rec.get('retries',0)+1}))
   continue
  out.parent.mkdir(parents=True,exist_ok=True)
  out.write_text(json.dumps({'record_key':key,'question_index':i,'answer':make_answer(q,aid),'model':MODEL,'raw_output':raw,'mode':'vllm_fp8_retry'}))
  err_out.unlink()
  fixed+=1
  print(f'[fixed] key={key} q={i} fixed={fixed} still_failed={still_failed}',flush=True)
  vol.commit()
 vol.commit()
 print(f'[retry-complete] fixed={fixed} still_failed={still_failed}',flush=True)
 return {'fixed':fixed,'still_failed':still_failed}
