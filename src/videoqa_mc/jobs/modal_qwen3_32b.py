"""Native-video Qwen3-VL ablation (smoke/full validation)."""
import json,re
from pathlib import Path
import modal
from videoqa_mc.mcqa_core import make_answer
ROOT=Path('/data/2702-multiple-choice-videoqa'); MODEL='Qwen/Qwen3-VL-32B-Instruct'
app=modal.App('unified-mcqa-2702-qwen3-32b')
vol=modal.Volume.from_name('perception-test-2026-data',create_if_missing=False)
img=(modal.Image.debian_slim(python_version='3.11').apt_install('ffmpeg').pip_install('accelerate>=1.9','qwen-vl-utils','torch==2.7.1','torchvision==0.22.1','transformers>=5.0','bitsandbytes').env({'HF_HOME':str(ROOT/'models/huggingface')}).add_local_python_source('videoqa_mc'))
def find_video(key,source):
 for root in (ROOT/'raw/validation',ROOT/'raw/full_validation'):
  for ident in (key,source):
   for p in root.rglob(f'{ident}*.mp4'):
    if p.stat().st_size>10000:return p
 raise FileNotFoundError(key)
def letter(s,n):
 m=re.search(r'\b([A-E])\b',s.upper())
 if not m or ord(m.group(1))-65>=n: raise ValueError(s)
 return ord(m.group(1))-65
def prompt(q):
 return 'Inspect the entire video, including temporal changes. Return exactly one option letter.\nQuestion: '+q['question']+'\nOptions:\n'+'\n'.join(f'{chr(65+i)}. {x}' for i,x in enumerate(q['options']))
@app.function(image=img,gpu='H100',volumes={'/data':vol},timeout=6*3600)
def answer_validation(limit:int=8,prefix:str='qwen3_32b'):
 limit=int(limit)
 import torch
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 from qwen_vl_utils import process_vision_info
 vol.reload();db=json.loads(next((ROOT/'manifests/validation_annotations').glob('*.json')).read_text())
 from transformers import BitsAndBytesConfig
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16)
 model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,quantization_config=quant,device_map='auto').eval();proc=AutoProcessor.from_pretrained(MODEL)
 done=0
 for key,r in db.items():
  try:v=find_video(key,str(r.get('metadata',{}).get('video_id','')))
  except FileNotFoundError:continue
  for i,q in enumerate(r.get('mc_question',[])):
   if done>=limit:return {'processed':done,'model':MODEL}
   out=ROOT/'predictions/validation'/f'{prefix}_{key}_{i}.json'
   if out.exists():continue
   msgs=[{'role':'user','content':[{'type':'video','video':str(v),'fps':0.1,'max_pixels':112*112},{'type':'text','text':prompt(q)}]}]
   text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
   image_inputs,video_inputs=process_vision_info(msgs)
   inputs=proc(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors='pt').to(model.device)
   with torch.inference_mode():gen=model.generate(**inputs,max_new_tokens=8)
   raw=proc.batch_decode(gen[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0].strip();a=letter(raw,len(q['options']))
   out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'record_key':key,'question_index':i,'answer':make_answer(q,a),'model':MODEL,'raw_output':raw,'mode':'native_video'}));done+=1
  vol.commit()
 vol.commit();return {'processed':done,'model':MODEL}
