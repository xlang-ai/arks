import os
import json
from collections import defaultdict
from arks.model import gpt_generate

def important_syntax(s):
    try:
        _ = float(s)
        return False
    except:
        pass
    return not ('def ' in s or s in ['result','def','for'] or s.startswith('#') or
                not any([c.isalnum() for c in s]) or 'import ' in s)

def find_code_explanations(problem_dir):
    map_line_to_explanation, _ = process_explanations(problem_dir)
    line_explanations_backup = []
    line_explanations = []
    for _, v in map_line_to_explanation.items():
        for d in v:
            if d['explanation'] not in line_explanations_backup:
                line_explanations_backup.append(d['explanation'])
                if important_syntax(d["syntax"]):
                    line_explanations.append(d['explanation'])
    if len(line_explanations) == 0:
        print("After filtering, there is no meaningful code explanations in",problem_dir)
        line_explanations = line_explanations_backup
    assert len(line_explanations)>0,f"{problem_dir}"
    return line_explanations

def process_explanations(problem_dir):
    with open(os.path.join(problem_dir,'code_explanation.json')) as f:
        try:
            raw_response = json.load(f)
            if isinstance(raw_response,list):
                explanations = raw_response
            else:
                raw_response = raw_response["choices"][0]["message"]["content"]
                explanations = json.loads(raw_response)
        except:
            raise ValueError(f"{problem_dir}")
    map_line_to_explanation = defaultdict(list)
    map_explanation_to_line = defaultdict(list)
    for d in explanations:
        for line_id in d["line ids"]:
            if not {'syntax': d["syntax"],'explanation': d["explanation"]} in map_line_to_explanation[line_id]:
                map_line_to_explanation[line_id].append({'syntax': d["syntax"],'explanation': d["explanation"]})
            if not line_id in map_explanation_to_line[f'{d["syntax"]}, {d["explanation"]}']:
                map_explanation_to_line[f'{d["syntax"]}, {d["explanation"]}'].append(line_id)
    return map_line_to_explanation,map_explanation_to_line

def find_error_code_explanations(problem_dir,find_error_lines_func):
    with open(os.path.join(problem_dir,'code_context.txt')) as f:
        code_context = f.read()
    assert code_context.count('[insert]')==1
    prefix = code_context.split('[insert]')[0]
    offset = prefix.count('\n')
    error_lines, _ = find_error_lines_func(problem_dir=problem_dir)
    map_line_to_explanation,_ = process_explanations(problem_dir)
    error_line_explanations = []
    for _,indices_lines in error_lines.items():
        for idx_line in indices_lines:
            cur_idx = idx_line[0]-offset
            if cur_idx in map_line_to_explanation:
                for d in map_line_to_explanation[cur_idx]:
                    if d['explanation'] not in error_line_explanations:
                        error_line_explanations.append(d['explanation'])
    if len(error_line_explanations)==0:
        error_line_explanations = find_code_explanations(problem_dir=problem_dir)
    return error_line_explanations

def prepare_explain_args(output_dir,output_tag,temperature):
    dict_str = '[{"syntax": ..., "explanation": ..., "line ids": [...]}, ...]'
    inst = f"Please explain the syntax used in each line.\n" \
            f"Please store the syntax explanation using the following json format:\n" \
            f"```json\n{dict_str}\n```"
    inference_args = []
    if not os.path.isfile(os.path.join(output_dir, 'syntax_correct_ids.json')):
        syntax_correct_ids = []
    else:
        with open(os.path.join(output_dir, 'syntax_correct_ids.json')) as f:
            syntax_correct_ids = json.load(f)
    for subdir in os.listdir(output_dir):
        if not subdir.startswith('q') or not os.path.isdir(os.path.join(output_dir,subdir)):
            continue
        if subdir in syntax_correct_ids and os.path.isfile(os.path.join(output_dir,subdir,'code_explanation.json')):
            continue
        with open(os.path.join(output_dir,subdir,'generated_code')) as f:
            program = f.read()
        program_lines = program.split('\n')
        processed_program_lines_short = []
        for line_id,line in enumerate(program_lines):
            if line_id<30:
                processed_program_lines_short.append(f'Line {line_id+1}: {line}')
            else:
                break
        processed_program = '\n'.join(processed_program_lines_short)
        prompt = f"{processed_program}\n\n{inst}"
        inference_args.append({
            'problem_dir': os.path.join(output_dir,subdir),
            'output_tag': output_tag,
            'model': "gpt-3.5-turbo-1106",
            'temperature': temperature,
            'messages': [{"role": "system", "content": "You are a good programmer"},{"role": "user", "content": prompt}],
            'mode': 'explanation'
        })
    return inference_args

def explain(output_dir,output_tag,num_worker=0):
    inference_args = prepare_explain_args(output_dir,output_tag,0)
    gpt_generate(
        configs=inference_args,
        num_worker=num_worker
    )
    for subdir in os.listdir(output_dir):
        if not subdir.startswith('q') or not os.path.isfile(os.path.join(output_dir,subdir,f"{output_tag}_explanation.json")):
            continue
        with open(os.path.join(output_dir,subdir,f"{output_tag}_explanation.json")) as f:
            cur_explanations = json.load(f)
        with open(os.path.join(output_dir,subdir,f"code_explanation.json"),'w') as f:
            json.dump(cur_explanations['explanations'],f,indent=2)


