"""Semantic final-answer leakage audit for the five teacher-strategy files."""
import argparse, json, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
MANIFEST=ROOT/'data'/'main'/'final_1050_problem_manifest.json'
STRATEGIES=ROOT/'data'/'teacher_strategies'
OUT=ROOT/'results'/'leakage'/'reruns'
MODEL='gpt-5.6-luna'
INSTRUCTIONS="""You are auditing teacher strategies for FINAL-ANSWER leakage. For each record, decide whether the teacher strategy gives the requested final answer. Count explicit answers, spelled-out answers, and mathematically equivalent forms (e.g. 4/6 for 2/3). Do NOT count a number merely because it appears in the problem, is a generic illustrative example, or is an intermediate quantity unless it establishes the requested final answer. Return one verdict for every record ID, and only JSON."""
SCHEMA={"type":"object","additionalProperties":False,"properties":{"results":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{"record_id":{"type":"string"},"answer_leakage":{"type":"boolean"},"form":{"type":"string","enum":["DIRECT","EQUIVALENT","NONE"]},"reason":{"type":"string"}},"required":["record_id","answer_leakage","form","reason"]}}},"required":["results"]}
def rows(x): return x if isinstance(x,list) else x.get('results') or x.get('records')
def load():
 manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))['questions']; answers={x['id']:x['reference_answer'] for x in manifest}; out=[]
 for p in sorted(STRATEGIES.glob('*.json')):
  for x in rows(json.loads(p.read_text(encoding='utf-8'))):
   out.append({'record_id':f'{p.name}::{x["global_index"]}','teacher_file':p.name,'global_index':x['global_index'],'question_id':x['id'],'problem':x.get('problem',''),'reference_answer':x.get('reference_answer') or answers[x['id']],'teacher_strategy':x['teacher_response']})
 return out
def seen(path):
 s=set()
 if path.exists():
  for line in path.read_text(encoding='utf-8').splitlines():
   try:
    x=json.loads(line)
    if x.get('status')=='success': s.update(r['record_id'] for r in x['results'])
   except: pass
 return s
def prompt(batch):
 return '\n\n'.join(f'<record id="{x["record_id"]}">\nProblem: {x["problem"]}\nReference answer: {x["reference_answer"]}\nTeacher strategy: {x["teacher_strategy"]}\n</record>' for x in batch)
def audit(batch):
 expected={x['record_id'] for x in batch}; client=OpenAI(default_headers={'Accept-Encoding':'identity'})
 for n in range(6):
  try:
   r=client.responses.create(model=MODEL,reasoning={'effort':'none'},instructions=INSTRUCTIONS,input=prompt(batch),text={'format':{'type':'json_schema','name':'leakage_verdict','strict':True,'schema':SCHEMA}},max_output_tokens=5000,store=False)
   result=json.loads(r.output_text)['results']
   if {x['record_id'] for x in result}!=expected: raise ValueError('missing record IDs')
   return {'status':'success','results':result}
  except Exception as e:
   if n==5:return {'status':'failed','record_ids':sorted(expected),'error':f'{type(e).__name__}: {e}'}
   time.sleep(min(60,2**n))
def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=10);p.add_argument('--chunk-size',type=int,default=20);a=p.parse_args()
 if not os.getenv('OPENAI_API_KEY'):raise RuntimeError('Set OPENAI_API_KEY')
 OUT.mkdir(parents=True,exist_ok=True); all_rows=load(); journal=OUT/'semantic_answer_leakage.jsonl'; pending=[x for x in all_rows if x['record_id'] not in seen(journal)]; batches=[pending[i:i+a.chunk_size] for i in range(0,len(pending),a.chunk_size)];print(f'{len(pending)} strategies remaining in {len(batches)} batches',flush=True)
 with journal.open('a',encoding='utf-8',buffering=1) as out,ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures=[pool.submit(audit,b) for b in batches]
  for i,f in enumerate(as_completed(futures),1):out.write(json.dumps(f.result(),ensure_ascii=False)+'\n');out.flush();print(f'{i}/{len(batches)} batches',flush=True)
 verdicts={}
 for line in journal.read_text(encoding='utf-8').splitlines():
  try:
   x=json.loads(line)
   if x.get('status')=='success':verdicts.update({r['record_id']:r for r in x['results']})
  except:pass
 report=[]
 for row in all_rows:
  v=verdicts.get(row['record_id']);
  if v:report.append({**{k:row[k] for k in ('record_id','teacher_file','global_index','question_id','reference_answer')},**v})
 (OUT/'semantic_answer_leakage_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 totals={}
 for f in sorted({x['teacher_file'] for x in report}):
  r=[x for x in report if x['teacher_file']==f]; totals[f]={'audited':len(r),'answer_leakage':sum(x['answer_leakage'] for x in r),'direct':sum(x['form']=='DIRECT' for x in r),'equivalent':sum(x['form']=='EQUIVALENT' for x in r)}
 (OUT/'semantic_answer_leakage_summary.json').write_text(json.dumps(totals,indent=2),encoding='utf-8')
if __name__=='__main__':main()
