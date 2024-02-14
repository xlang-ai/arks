import os
import json
import shutil
import argparse
import subprocess
import multiprocessing as mp
from tqdm import tqdm
from arks.utils import check_dir_txt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def Pony_worker(arguments):
    cur_dir,turn=arguments
    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cur_dir, 'logs')):
        shutil.rmtree(os.path.join(cur_dir, 'logs'))
    os.makedirs(os.path.join(cur_dir, 'logs') ,exist_ok=True)
    if os.path.isdir(os.path.join(cur_dir, 'logs2')):
        shutil.rmtree(os.path.join(cur_dir, 'logs2'))
    os.makedirs(os.path.join(cur_dir, 'logs2') ,exist_ok=True)
    if turn > -1:
        while not os.path.isfile(os.path.join(cur_dir, f"{turn}_generation.json")):
            turn -= 1
        with open(os.path.join(cur_dir, f"{turn}_generation.json")) as f:
            cur_response = json.load(f)
        generated_code = cur_response['generated_code']
        try:
            with open(os.path.join(cur_dir, 'generated_code'),'w') as f:
                f.write(generated_code)
        except:
            pass
    else:
        try:
            with open(os.path.join(cur_dir, 'generated_code')) as f:
                generated_code = f.read()
        except:
            pass
    with open(os.path.join(cur_dir, 'code_context.txt')) as f:
        code_context = f.read()
    assert code_context.count("[insert]")==1 
    program = code_context.replace("[insert]", generated_code)
    if not os.path.isdir(os.path.join(cur_dir, "program")):
        os.makedirs(os.path.join(cur_dir, "program"),exist_ok=True)
    with open(os.path.join(cur_dir, "program",'main.pony'), "w", encoding="UTF-8") as f:
        f.write(program)
    if os.path.isdir(os.path.join(cur_dir, "outputs")):
        shutil.rmtree(os.path.join(cur_dir, "outputs"))
    os.makedirs(os.path.join(cur_dir, "outputs"))
    os.chdir(cur_dir)

    test_case = 0
    while True:
        if os.path.isfile('input.txt'):
            os.remove('input.txt')
        if os.path.isfile('program1'):
            os.remove('program1')
        test_case += 1
        if not os.path.isfile(f'inputs/input{test_case}.txt') or not os.path.isfile(f'ans/ans{test_case}.txt'):
            break
        shutil.copyfile(f'inputs/input{test_case}.txt','input.txt')
        complile_cmd = "ponyc program"
        with open(os.path.join('logs',f"{test_case}.txt"), "w") as error_log:
            try:
                subprocess.run(complile_cmd, stderr=error_log, shell=True, timeout=10,text=True)
            except subprocess.TimeoutExpired as e:
                error_log.write("Execution timed out after 60 seconds")
        exe_cmd = f"./program1 > outputs/ans{test_case}.txt"
        with open(os.path.join('logs2',f"{test_case}.txt"), "w") as error_log:
            try:
                subprocess.run(exe_cmd, stderr=error_log, shell=True, timeout=10,text=True)
            except subprocess.TimeoutExpired as e:
                error_log.write("Execution timed out after 60 seconds")
        try:
            with open(os.path.join('logs',f"{test_case}.txt")) as log_file:
                cur_log = log_file.read().strip()
        except:
            print(cur_dir,test_case)
            exit(0)
        if 'Linking ./program1' in cur_log:
            assert 'Writing ./program.o' in cur_log
            os.remove(os.path.join('logs',f"{test_case}.txt"))

    if not check_dir_txt('logs'):
        shutil.rmtree('logs')
    if os.path.isdir('logs'):
        with_log = True
    else:
        with_log = False

    correct = True
    with_result = True
    for file in os.listdir('ans'):
        if not (file.startswith('ans') and file.endswith('.txt')):
            continue
        if not os.path.isfile(os.path.join('outputs',file)):
            correct = False
            with_result = False
            break
        with open(os.path.join('ans',file)) as cur_file:
            gold = cur_file.read().strip()
        with open(os.path.join('outputs',file)) as cur_file:
            pred = cur_file.read().strip()
        try:
            gold = round(float(gold),2)
            pred = round(float(pred),2)
        except:
            pass
        if pred!=gold:
            correct = False
            break
    os.chdir(cwd)
    return correct,with_log,with_result,cur_dir

def Pony_eval(output_dir,num_workers=-1,turn=-1):
    directories = [(os.path.join(output_dir, d),turn) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir,d)) and d.startswith('q')]
    correct = 0
    total = len(directories)
    with_log = 0
    with_result = 0
    syntax_incorrect = []
    syntax_correct = []
    correct_examples = []
    if num_workers==-1:
        num_workers = mp.cpu_count()//2
    all_results = []
    if num_workers>1:
        with mp.Pool(num_workers) as pool, tqdm(total=len(directories), desc='Evaluate') as pbar:
            for result in pool.imap_unordered(Pony_worker, directories):
                pbar.update()
                all_results.append(result)
    else:
        for cur_dir in tqdm(directories):
            result = Pony_worker(cur_dir)
            all_results.append(result)
    for result in all_results:
        if result[0]:
            correct += 1
            correct_examples.append(result[-1].split('/')[-1])
        if result[1]:
            with_log += 1
        if result[1]:
            syntax_incorrect.append(result[-1].split('/')[-1])
        else:
            syntax_correct.append(result[-1].split('/')[-1])
        if result[2]:
            with_result += 1
    print('Accuracy:', correct / total)
    print("Syntax accuracy by log:", 1 - len(syntax_incorrect) / total)
    print('syntax incorrect examples:',sorted(syntax_incorrect))
    print('correct examples:',sorted(correct_examples))
    with open(os.path.join(output_dir,f"result_{args.turn}.json"),'w') as f:
        json.dump({
            "Accuracy": correct / total,
            "Syntax accuracy": 1 - with_log / total,
            "Examples with incorrect syntax": sorted(syntax_incorrect),
            "Correct examples": sorted(correct_examples)
        },f,indent=2)
    with open(os.path.join(output_dir,'syntax_correct_ids.json'),'w') as f:
        json.dump(syntax_correct,f,indent=2)
    with open(os.path.join(output_dir,'correct_ids.json'),'w') as f:
        json.dump(correct_examples,f,indent=2)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",type=str)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--turn", type=int, default=-1)
    args = parser.parse_args()
    Pony_eval(output_dir=args.output_dir,num_workers=args.num_workers,turn=args.turn)


