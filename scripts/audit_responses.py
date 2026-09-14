"""Fast bulk correctness audit from audit_zip.zip.

Writes one small per-problem label JSON for each replicate and an aggregate
summary.  Each label includes the exact source file / run / student / condition.
"""
from __future__ import annotations
import argparse, json, os, re, time, zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, stdev
from openai import OpenAI

MAX_WORKERS = 40
DEFAULT_MODEL = "gpt-5.6-luna"
INSTRUCTIONS = """Audit each math response below. Return CORRECT if it reaches a mathematically correct requested answer (boxed form is not required; equivalent forms and valid alternative methods count), INCORRECT if it does not, or UNCERTAIN only when the stored response truly cannot be determined. Return exactly one result for every record_id and no explanation."""
SCHEMA = {"type":"object","additionalProperties":False,"properties":{"results":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{"record_id":{"type":"string"},"correctness":{"type":"string","enum":["CORRECT","INCORRECT","UNCERTAIN"]}},"required":["record_id","correctness"]}}},"required":["results"]}

def value(row, *keys):
    for key in keys:
        if row.get(key) is not None: return str(row[key])
    return ""

def load_records(zip_path):
    records=[]
    with zipfile.ZipFile(zip_path) as z:
        names=[n for n in z.namelist() if re.search(r"(?:^|/)run[123]_.*\.json$", n, re.I)]
        if len(names) != 36: raise RuntimeError(f"Expected 36 run JSON files in ZIP; found {len(names)}.")
        for name in sorted(names):
            data=json.loads(z.read(name).decode("utf-8")); rows=data if isinstance(data,list) else data.get("results")
            if not isinstance(rows,list) or len(rows)!=1050: raise RuntimeError(f"{name}: expected 1050 records")
            source=Path(name).name; match=re.match(r"run(\d)_",source,re.I)
            for row in rows:
                index=int(row["global_index"])
                records.append({"record_id":f"{source}::{index}","source_file":source,"run":value(row,"run") or match.group(1),"student":value(row,"student","student_model"),"condition":value(row,"teacher","condition") or "Normal","global_index":index,"question_id":value(row,"question_id","id"),"problem":value(row,"problem","question"),"reference_answer":value(row,"reference_answer","answer"),"student_response":value(row,"student_response","response","output")})
    return records

def chunks(rows,size): return [rows[i:i+size] for i in range(0,len(rows),size)]
def prompt(batch):
    return "\n\n".join(f'<record id="{r["record_id"]}">\nProblem: {r["problem"]}\nReference answer: {r["reference_answer"]}\nStudent response: {r["student_response"]}\n</record>' for r in batch)

def prior_ids(path):
    ids=set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item=json.loads(line)
                if item.get("status")=="success": ids.update(x["record_id"] for x in item["results"])
            except (json.JSONDecodeError,KeyError): pass
    return ids

def one_batch(batch, model, replicate):
    expected={r["record_id"] for r in batch}; client=OpenAI(default_headers={"Accept-Encoding":"identity"})
    for attempt in range(6):
        try:
            response=client.responses.create(model=model, reasoning={"effort":"none"}, instructions=INSTRUCTIONS, input=prompt(batch), text={"format":{"type":"json_schema","name":"audit_labels","strict":True,"schema":SCHEMA}}, max_output_tokens=12000, store=False)
            results=json.loads(response.output_text)["results"]
            if {x["record_id"] for x in results} != expected: raise ValueError("missing or unexpected IDs")
            return {"replicate":replicate,"status":"success","results":results}
        except Exception as error:
            if attempt==5: return {"replicate":replicate,"status":"failed","record_ids":sorted(expected),"error":f"{type(error).__name__}: {error}"}
            time.sleep(min(60,2**attempt))

def materialize(base, records, replicates, model):
    metadata={r["record_id"]:{k:r[k] for k in ("record_id","source_file","run","student","condition","global_index","question_id")} for r in records}; report={"model":model,"concurrency":MAX_WORKERS,"replicates":[]}
    for rep in range(1,replicates+1):
        labels={}; journal=base/f"bulk_labels_replicate_{rep:02d}.jsonl"
        if journal.exists():
            for line in journal.read_text(encoding="utf-8").splitlines():
                try:
                    item=json.loads(line)
                    if item.get("status")=="success": labels.update({x["record_id"]:x["correctness"] for x in item["results"]})
                except json.JSONDecodeError: pass
        rows=[{**metadata[key],"correctness":label} for key,label in sorted(labels.items())]
        (base/f"bulk_labels_replicate_{rep:02d}.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
        conditions={}
        for file in sorted({r["source_file"] for r in rows}):
            group=[r for r in rows if r["source_file"]==file]; c=Counter(r["correctness"] for r in group)
            conditions[file]={"run":group[0]["run"],"student":group[0]["student"],"condition":group[0]["condition"],"total":len(group),"correct":c["CORRECT"],"incorrect":c["INCORRECT"],"uncertain":c["UNCERTAIN"],"accuracy":c["CORRECT"]/len(group) if group else None}
        total=Counter(r["correctness"] for r in rows); report["replicates"].append({"replicate":rep,"total":len(rows),"correct":total["CORRECT"],"incorrect":total["INCORRECT"],"uncertain":total["UNCERTAIN"],"accuracy":total["CORRECT"]/len(rows) if rows else None,"by_condition_file":conditions})
    accuracies=[r["accuracy"] for r in report["replicates"] if r["accuracy"] is not None]; report["accuracy_mean"]=mean(accuracies) if accuracies else None; report["accuracy_sample_sd"]=stdev(accuracies) if len(accuracies)>1 else None
    (base/"bulk_labels_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--replicates",type=int,default=3); parser.add_argument("--chunk-size",type=int,default=200); parser.add_argument("--workers",type=int,default=40); parser.add_argument("--model",default=DEFAULT_MODEL); parser.add_argument("--limit",type=int); parser.add_argument("--input-zip",type=Path); parser.add_argument("--output-dir",type=Path); args=parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"): raise RuntimeError("Set OPENAI_API_KEY before running.")
    if not 1<=args.workers<=MAX_WORKERS: raise ValueError("workers must be between 1 and 40")
    project_root=Path(__file__).resolve().parents[1]
    input_zip=args.input_zip or project_root/"data"/"main"/"audit_input_bundle.zip"
    base=args.output_dir or project_root/"results"/"main"/"audits"/"reruns"
    base.mkdir(parents=True,exist_ok=True)
    records=load_records(input_zip)
    if args.limit is not None: records=records[:args.limit]
    print(f"Loaded {len(records)} records from 36 conditions.")
    for rep in range(1,args.replicates+1):
        journal=base/f"bulk_labels_replicate_{rep:02d}.jsonl"; seen=prior_ids(journal); pending=[r for r in records if r["record_id"] not in seen]; work=chunks(pending,args.chunk_size); print(f"Replicate {rep}: {len(pending)} records / {len(work)} batches remaining")
        with journal.open("a",encoding="utf-8",buffering=1) as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(one_batch,b,args.model,rep) for b in work]
            for n,future in enumerate(as_completed(futures),1):
                out.write(json.dumps(future.result(),ensure_ascii=False)+"\n"); out.flush()
                if n%10==0 or n==len(work): print(f"Replicate {rep}: {n}/{len(work)} batches finished")
    materialize(base,records,args.replicates,args.model); print("Finished: three label JSONs + bulk_labels_summary.json")
if __name__=="__main__": main()
