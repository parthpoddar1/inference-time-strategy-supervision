import json, glob, re, os
from fractions import Fraction
from pathlib import Path

BASE=Path(r'C:\Users\Parth Poddar\Desktop\audit')
questions=json.loads((BASE/'final_manifest.json').read_text(encoding='utf-8'))['questions']
answers={q['id']:q['reference_answer'] for q in questions}

ONES=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
TENS=['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
def words(n):
    if n<20:return ONES[n]
    if n<100:return TENS[n//10]+((' '+ONES[n%10]) if n%10 else '')
    if n<1000:return ONES[n//100]+' hundred'+((' '+words(n%100)) if n%100 else '')
    return ''
def parse_answer(a):
    s=str(a).replace('\\frac{','').replace('}{','/').replace('}','').replace('\\$','$').replace('\\%','%').replace(',','')
    m=re.search(r'(-?\d+(?:\.\d+)?\s*/\s*-?\d+(?:\.\d+)?)|(-?\d+(?:\.\d+)?)',s)
    if not m:return None
    try:return Fraction(m.group(0).replace(' ','')) / (100 if '%' in s else 1)
    except: return None
def rows(x):return x if isinstance(x,list) else x.get('results') or x.get('records')
def values_in(text):
    vals=[]
    for m in re.finditer(r'(?<![A-Za-z0-9.])-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?(?![A-Za-z0-9.])',text):
        try:vals.append(Fraction(m.group(0).replace(' ','')))
        except:pass
    return vals
for p in glob.glob(str(BASE/'teacher_strategy'/'*.json')):
    data=rows(json.loads(Path(p).read_text(encoding='utf-8'))); numeric=[]; word=[]
    for r in data:
        target=parse_answer(r.get('reference_answer') or answers[r['id']]); text=r['teacher_response'].lower()
        if target is None:continue
        numeric_hit=target in values_in(text)
        nword=False
        if target.denominator==1 and 0<=target.numerator<1000:
            w=words(target.numerator)
            if w:nword=bool(re.search(r'(?<![a-z])'+re.escape(w).replace(r'\ ',r'[ -]')+r'(?![a-z])',text))
        if numeric_hit:numeric.append(r['global_index'])
        if nword:word.append(r['global_index'])
    union=sorted(set(numeric)|set(word))
    print(os.path.basename(p), 'numeric_equivalent',len(numeric),'number_word',len(word),'union',len(union))
