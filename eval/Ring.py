import os
import sys
import json
import shutil
import argparse
import subprocess
import multiprocessing as mp
from tqdm import tqdm
from arks.utils import check_dir_txt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def Ring_worker(arguments):
    cur_dir,turn=arguments
    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cur_dir, 'logs')):
        shutil.rmtree(os.path.join(cur_dir, 'logs'))
    os.makedirs(os.path.join(cur_dir, 'logs') ,exist_ok=True)
    if turn > -1:
        while not os.path.isfile(os.path.join(cur_dir, f"{turn}_generation.json")):
            turn -= 1
        with open(os.path.join(cur_dir, f"{turn}_generation.json")) as f:
            cur_response = json.load(f)
        generated_code = cur_response['generated_code']
        with open(os.path.join(cur_dir, 'generated_code'),'w') as f:
            f.write(generated_code)
    else:
        with open(os.path.join(cur_dir, 'generated_code')) as f:
            generated_code = f.read()
    with open(os.path.join(cur_dir, 'code_context.txt')) as f:
        code_context = f.read()
    assert code_context.count("[insert]")==1 and code_context.count("\n[insert]")==1
    program = code_context.replace("[insert]", generated_code)
    with open(os.path.join(cur_dir, "program.ring"), "w", encoding="UTF-8") as f:
        f.write(program)
    if os.path.isdir(os.path.join(cur_dir, "outputs")):
        shutil.rmtree(os.path.join(cur_dir, "outputs"))
    os.makedirs(os.path.join(cur_dir, "outputs"))
    if os.path.isdir(os.path.join(cur_dir, "results")):
        shutil.rmtree(os.path.join(cur_dir, "results"))
    os.makedirs(os.path.join(cur_dir, "results"))
    os.chdir(cur_dir)

    test_case = 0
    if not os.path.isdir('e_outputs'):
        os.makedirs('e_outputs', exist_ok=True)
    correct = True
    with_result = True
    while True:
        if os.path.isfile(os.path.join('outputs', 'output.txt')):
            os.remove(os.path.join('outputs', 'output.txt'))
        test_case += 1
        if not os.path.isfile(f'inputs/input{test_case}.txt') or not os.path.isfile(f'ans/ans{test_case}.txt'):
            break
        my_cmd = f"ring program.ring < inputs/input{test_case}.txt > logs/{test_case}.txt"
        with open(os.path.join('e_outputs',f"{test_case}.txt"), "w") as output_log:
            try:
                subprocess.run(my_cmd, stderr=output_log, shell=True, timeout=60,text=True)
            except subprocess.TimeoutExpired as e:
                output_log.write("Execution timed out after 60 seconds")
        with open(os.path.join('logs',f"{test_case}.txt")) as log_file:
            cur_log = log_file.read().strip()
        if len(cur_log)==0:
            os.remove(os.path.join('logs',f"{test_case}.txt"))
        if os.path.isfile(os.path.join('outputs', 'output.txt')):
            shutil.copyfile(os.path.join('outputs', 'output.txt'),os.path.join('results', f'ans{test_case}.txt'))

    # for file in os.listdir('ans'):
    #     if not (file.startswith('ans') and file.endswith('.txt')):
    #         continue
        file = f"ans{test_case}.txt"
        if not os.path.isfile(os.path.join('results',file)):
            correct = False
            with_result = False
            break
        with open(os.path.join('ans',file)) as cur_file:
            gold = cur_file.read()
        if gold == 'false':
            gold = 0
        elif gold == 'true':
            gold = 1
        with open(os.path.join('results',file)) as cur_file:
            pred = cur_file.read()
        rel_error = 1
        try:
            gold = round(float(gold), 6)
            pred = round(float(pred), 6)
            rel_error = (pred - gold) / gold
        except:
            pass
        if pred != gold and rel_error > 1e-6:
            correct = False
            break
    if not check_dir_txt('logs'):
        shutil.rmtree('logs')
    if os.path.isdir('logs'):
        with_log = True
    else:
        with_log = False
    os.chdir(cwd)
    return correct,with_log,with_result,cur_dir

def Ring_eval(output_dir,num_workers=-1,turn=-1):
    directories = [(os.path.join(output_dir, d),turn) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir,d)) and d.startswith('q')]
    correct = 0
    total = len(directories)
    with_log = 0
    with_result = 0
    syntax_incorrect = []
    syntax_correct = []
    correct_examples = []
    if num_workers==-1:
        num_workers = mp.cpu_count() // 4
    all_results = []
    if num_workers>1:
        with mp.Pool(num_workers) as pool, tqdm(total=len(directories), desc='Evaluate') as pbar:
            for result in pool.imap_unordered(Ring_worker, directories):
                pbar.update()
                all_results.append(result)
    else:
        for cur_dir in tqdm(directories):
            result = Ring_worker(cur_dir)
            all_results.append(result)
    for result in all_results:
        if result[0]:
            correct += 1
            correct_examples.append(result[-1].split('/')[-1])
        if result[1]:
            with_log += 1
        if not result[2]:
            syntax_incorrect.append(result[-1].split('/')[-1])
        else:
            syntax_correct.append(result[-1].split('/')[-1])
        if result[2]:
            with_result += 1
        # if result[1] and result[2]:
        #     print(result[3], 'have both')
        # if not result[1] and not result[2]:
        #     print(result[3], 'has neither')
    print('Accuracy:', correct / total)
    print("Syntax accuracy by log:", 1 - len(syntax_incorrect) / total)
    # print("Syntax accuracy by output:", with_result / total)
    print('syntax incorrect examples:',sorted(syntax_incorrect))
    print('correct examples:',sorted(correct_examples))
    with open(os.path.join(output_dir,f"result_{args.turn}.json"),'w') as f:
        json.dump({
            "Accuracy": correct / total,
            "Syntax accuracy": 1 - len(syntax_incorrect) / total,
            "Examples with incorrect syntax": sorted(syntax_incorrect),
            "Correct examples": sorted(correct_examples)
        },f,indent=2)
    with open(os.path.join(output_dir,'syntax_correct_ids.json'),'w') as f:
        json.dump(syntax_correct,f)
    with open(os.path.join(output_dir,'correct_ids.json'),'w') as f:
        json.dump(correct_examples,f,indent=2)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",type=str)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--turn", type=int, default=-1)
    args = parser.parse_args()
    Ring_eval(output_dir=args.output_dir,num_workers=args.num_workers,turn=args.turn)


