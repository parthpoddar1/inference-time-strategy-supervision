import json, os, re, time, zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

BASE=Path(__file__).resolve().parent; ROOT=BASE.parent; ZIP=ROOT/'data'/'main'/'audit_input_bundle.zip'; RESULTS=ROOT/'results'/'leakage'; OUT=RESULTS/'reruns'; MODEL='gpt-5.6-luna'
MAP={'DeepSeekV4Flash_teacher_strategies_1050.json':'deepseek','gemma_27b_teacher_strategies_1050.json':'gemma','Llama3.3_70b_teacher_strategies_1050.json':'llama','mistral_3.2_24b_teacher_strategies_1050.json':'mistral','Qwen72b_teacher_strategies_1050.json':'qwen','deepseek_v4_flash_0423_strategies.json':'deepseek','gemma_3_27b_it_strategies.json':'gemma','llama_3_3_70b_instruct_strategies.json':'llama','mistral_small_3_2_24b_instruct_strategies.json':'mistral','qwen_2_5_72b_instruct_strategies.json':'qwen'}
INST='''Audit each math student response. Return CORRECT only if it establishes the requested correct answer; accept unboxed, spelled-out, and equivalent forms. Return INCORRECT otherwise. Return exactly one result per record ID and only JSON.'''
SCHEMA={'type':'object','additionalProperties':False,'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,'properties':{'record_id':{'type':'string'},'correctness':{'type':'string','enum':['CORRECT','INCORRECT']}},'required':['record_id','correctness']}}},'required':['results']}
def leaked():
 d=json.loads((RESULTS/'semantic_answer_leakage_results.json').read_text(encoding='utf-8')); out=defaultdict(set)
 for x in d:
  if x['answer_leakage']:out[MAP[x['teacher_file']]].add(x['global_index'])
 return out
def load():
 leak=leaked();out=[]
 with zipfile.ZipFile(ZIP) as z:
  for name in z.namelist():
   m=re.search(r'run([123])_(llama|qwen)_(qwen|deepseek|llama|gemma|mistral)\.json$',name,re.I)
   if not m:continue
   run,student,teacher=m.groups();data=json.loads(z.read(name).decode('utf-8')); rows=data if isinstance(data,list) else data['results']
   for r in rows:
    if r['global_index'] in leak[teacher.lower()]:out.append({'record_id':f'{Path(name).name}::{r["global_index"]}','run':int(run),'student':student.lower(),'teacher':teacher.lower(),'global_index':r['global_index'],'problem':r.get('problem',''),'reference_answer':r.get('reference_answer',''),'student_response':r.get('student_response',r.get('response',r.get('output','')))})
 return out
def seen(p):
 s=set()
 if p.exists():
  for l in p.read_text(encoding='utf-8').splitlines():
   try:
    x=json.loads(l)
    if x.get('status')=='success':s.update(r['record_id'] for r in x['results'])
   except:pass
 return s
def one(batch):
 ids={x['record_id'] for x in batch}; text='\n\n'.join(f'<record id="{x["record_id"]}">Problem: {x["problem"]}\nReference answer: {x["reference_answer"]}\nStudent response: {x["student_response"]}</record>' for x in batch);client=OpenAI(default_headers={'Accept-Encoding':'identity'})
 for i in range(6):
  try:
   r=client.responses.create(model=MODEL,reasoning={'effort':'none'},instructions=INST,input=text,text={'format':{'type':'json_schema','name':'labels','strict':True,'schema':SCHEMA}},max_output_tokens=3000,store=False);a=json.loads(r.output_text)['results']
   if {x['record_id'] for x in a}!=ids:raise ValueError('missing IDs')
   return {'status':'success','results':a}
  except Exception as e:
   if i==5:return {'status':'failed','record_ids':list(ids),'error':str(e)}
   time.sleep(2**i)
def main():
 if not os.getenv('OPENAI_API_KEY'):raise RuntimeError('Set OPENAI_API_KEY')
 OUT.mkdir(parents=True,exist_ok=True); allrows=load();j=OUT/'leakage_case_accuracy.jsonl';pending=[x for x in allrows if x['record_id'] not in seen(j)];bs=[pending[i:i+20] for i in range(0,len(pending),20)];print(len(allrows),len(pending),len(bs),flush=True)
 with j.open('a',encoding='utf-8',buffering=1) as f,ThreadPoolExecutor(max_workers=8) as pool:
  fs=[pool.submit(one,b) for b in bs]
  for n,x in enumerate(as_completed(fs),1):f.write(json.dumps(x.result())+'\n');f.flush();print(n,len(bs),flush=True)
 labels={}
 for l in j.read_text(encoding='utf-8').splitlines():
  try:
   x=json.loads(l)
   if x.get('status')=='success':labels.update({r['record_id']:r['correctness'] for r in x['results']})
  except:pass
 rows=[{**{k:x[k] for k in ('record_id','run','student','teacher','global_index')},'correctness':labels[x['record_id']]} for x in allrows if x['record_id'] in labels]
 (OUT/'leakage_case_accuracy_results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
 g=defaultdict(list)
 for x in rows:g[x['teacher']].append(x)
 summary={t:{'audited':len(x),'correct':sum(z['correctness']=='CORRECT' for z in x),'accuracy':sum(z['correctness']=='CORRECT' for z in x)/len(x)} for t,x in g.items()}
 (OUT/'leakage_case_accuracy_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
if __name__=='__main__':main()
