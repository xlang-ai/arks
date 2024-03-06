import json
import argparse
import os.path

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str)
parser.add_argument("--lib", type=str)
args = parser.parse_args()

total = 0
correct_include = 0
lib = args.lib
for subdir in os.listdir(args.output_dir):
    if not subdir.startswith('q') or not os.path.isdir(os.path.join(args.output_dir,subdir)):
        continue
    with open(f"arks_data/Pony/examples/{subdir}/documentation.txt") as f:
        gold_doc = f.read()
    if lib=='Ring':
        gold_doc_parts = gold_doc.split('Document')
        gold_doc_parts1 = []
        for p in gold_doc_parts:
            if 'example' in p.lower():
                gold_doc_parts1.append(p[:p.lower().index('example')])
        gold_doc = '\n'.join(gold_doc_parts1)
    elif lib=='Pony':
        if '```' in gold_doc:
            import re
            indices = [0]
            indices += [m.start() for m in re.finditer('```', gold_doc)]
            gold_doc1 = ''
            for i in range(0, len(indices) - 1, 2):
                gold_doc1 += gold_doc[indices[i]:indices[i + 1]].replace('```', '')
            gold_doc = gold_doc1
    gold_doc_lines = gold_doc.split('\n')
    turn = 25
    while turn>0:
        if os.path.isfile(os.path.join(args.output_dir,f"{subdir}",f"{turn}_generation_config.json")):
            break
        turn -= 1
    if turn==0:
        continue
    with open(os.path.join(args.output_dir,f"{subdir}",f"{turn}_generation_config.json")) as f:
        c = json.load(f)
    include_count = 0
    cur_num = 0
    for l in gold_doc_lines:
        if 'Document' in l:
            continue
        cur_num += 1
        if l.strip() in c['messages'][1]['content']:
            include_count += 1
    if include_count>cur_num*0.7:
        correct_include += 1
    total += 1
print(correct_include,total,correct_include/total)
