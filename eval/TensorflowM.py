import os
import sys
import json
import ast
import pickle
import shutil
import argparse
import subprocess
import warnings
import multiprocessing as mp
from tqdm import tqdm
from arks.utils import check_dir_txt,import_source_file
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def Tensorflow_modified_easy_worker(arguments):
    cur_dir, turn, show_error_source = arguments
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
    else:
        with open(os.path.join(cur_dir, 'generated_code')) as f:
            generated_code = f.read()
    with open(os.path.join(cur_dir, 'code_context.txt')) as f:
        code_context = f.read()
    assert code_context.count("[insert]")==1 and code_context.count("\n[insert]\n")==1
    program = code_context.replace("[insert]", generated_code)
    with open(os.path.join(cur_dir, "program.py"), "w", encoding="UTF-8") as f:
        f.write(program)
    if os.path.exists(os.path.join(cur_dir, "result")):
        shutil.rmtree(os.path.join(cur_dir, "result"))
    os.mkdir(os.path.join(cur_dir, "result"))
    os.chdir(cur_dir)

    test_case = 0
    if not os.path.isdir('e_outputs'):
        os.makedirs('e_outputs', exist_ok=True)
    while True:
        test_case += 1
        if not os.path.isfile(f'input/input{test_case}.pkl') or not os.path.isfile(f'ans/ans{test_case}.pkl'):
            break
        my_cmd = f"python program.py --test_case {test_case} > e_outputs/out{test_case}"
        with open(os.path.join('logs',f"{test_case}.txt"), "w") as error_log:
            try:
                execution_result = subprocess.run(my_cmd, stderr=error_log, shell=True, timeout=60 ,text=True)
                execution_return_code = execution_result.returncode
            except subprocess.TimeoutExpired as e:
                execution_return_code = 1
                error_log.write("Execution timed out after 60 seconds")
        if execution_return_code==0:
            os.remove(os.path.join('logs',f"{test_case}.txt"))
    if not check_dir_txt('logs'):
        shutil.rmtree('logs')
    if os.path.isdir('logs'):
        with_log = True
    else:
        with_log = False

    test_module = import_source_file("test_code.py", "test_code")

    pass_flag = True
    with open(os.path.join("test_code.py")) as f:
        test_code_script = f.read()

    if '\ndef stringTest(' in test_code_script:
        generated_code_lines = generated_code.split("\n")
        processed_generated_code_lines = []
        for line in generated_code_lines:
            if "print" in line and "#" not in line.split("print"):
                continue
            else:
                processed_generated_code_lines.append(line)
        generated_code = "\n".join(processed_generated_code_lines)
        try:
            pass_flag = test_module.stringTest(generated_code)
            if not pass_flag and show_error_source:
                print(cur_dir.split('/')[-1],1)
        except:
            pass_flag = False
            if show_error_source:
                print(cur_dir.split('/')[-1],2)

    for i in range(1, test_case):
        if not pass_flag:
            break
        if not os.path.exists(f"result/result_{i}.pkl"):
            pass_flag = False
            if show_error_source:
                print(cur_dir.split('/')[-1],3)
        else:
            try:
                result = pickle.load(open(f"result/result_{i}.pkl", "rb"))
                try:
                    expected_result = pickle.load(open(f"ans/ans{i}.pkl", "rb"))
                    pass_flag = test_module.test(result, expected_result) == 1
                    if not pass_flag and show_error_source:
                        print(cur_dir.split('/')[-1],4)
                except:
                    pass_flag = False
                    if show_error_source:
                        print(cur_dir.split('/')[-1],5)
            except:
                pass_flag = False
                if show_error_source:
                    print(cur_dir.split('/')[-1],6)
    if os.path.isfile(f"result/result_1.pkl"):
        with_result = True
    else:
        with_result = False
    os.chdir(cwd)
    return pass_flag,with_log,with_result,cur_dir

def Tensorflow_modified_easy_eval(output_dir,turn=-1,num_workers=-1,show_error_source=False):
    directories = [(os.path.join(output_dir, d),turn,show_error_source) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir,d)) and d.startswith('q')]
    correct = 0
    total = len(directories)
    with_log = 0
    with_result = 0
    syntax_incorrect = []
    syntax_correct = []
    correct_examples = []
    if num_workers==-1:
        num_workers = mp.cpu_count()//2
    with mp.Pool(num_workers) as pool, tqdm(total=len(directories), desc='Evaluate') as pbar:
        for result in pool.imap_unordered(Tensorflow_modified_easy_worker, directories):
            pbar.update()

    # for d in tqdm(directories):
    #     result = Tensorflow_modified_easy_worker(d)
            if result[0]:
                correct += 1
                correct_examples.append(result[-1].split('/')[-1])
            if result[1]:
                with_log += 1
                syntax_incorrect.append(result[-1].split('/')[-1])
            else:
                syntax_correct.append(result[-1].split('/')[-1])
            if result[2]:
                with_result += 1


    print('Accuracy:', correct / total)
    print("Syntax accuracy:", 1 - with_log / total)
    # print("Syntax accuracy by output:", with_result / total)
    print("Examples with incorrect syntax:", sorted(syntax_incorrect))
    print("Correct examples", sorted(correct_examples))
    with open(os.path.join(output_dir,f"result_{args.turn}.txt"),'w') as f:
        f.write(f"Accuracy: {correct / total}")
        f.write(f"Syntax accuracy: {1 - with_log / total}")
        f.write(f"Examples with incorrect syntax: {sorted(syntax_incorrect)}")
        f.write(f"Correct examples: {sorted(correct_examples)}")
    with open(os.path.join(output_dir,'syntax_correct_ids.json'),'w') as f:
        json.dump(syntax_correct,f,indent=2)
    with open(os.path.join(output_dir,'correct_ids.json'),'w') as f:
        json.dump(correct_examples,f,indent=2)

if __name__=='__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",type=str)
    parser.add_argument("--turn", type=int,default=-1)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--show_error_source", action='store_true')
    args = parser.parse_args()
    Tensorflow_modified_easy_eval(output_dir=args.output_dir,num_workers=args.num_workers,turn=args.turn,
                                  show_error_source=args.show_error_source)


