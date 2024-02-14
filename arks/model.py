import os
import re
import time
import json
import openai
import random
import multiprocessing as mp
from tqdm import tqdm,trange
from arks.utils import clean_str,extract_program

LANGUAGE_MAP = {
    'Pony': 'pony',
    "Ring": "ring",
    "ScipyM": "python",
    "TensorflowM": "python"
}

def gpt_worker(config):
    for i in range(len(config['messages'])):
        config['messages'][i]["content"] = clean_str(config['messages'][i]["content"])
    if os.path.isfile(os.path.join(config['problem_dir'],f"{config['output_tag']}_{config['mode']}.json")):
        return None
    response = None
    execution_count = 0
    seed = 42
    temperature = config['temperature']
    while response is None and execution_count<10:
        try:
            execution_count += 1
            response = openai.ChatCompletion.create(model=config['model'],
                                                    temperature=temperature,
                                                    messages=config['messages'],
                                                    top_p=0,
                                                    request_timeout=30,
                                                    seed=seed)
            if 'sorry' in response["choices"][0]["message"]["content"].lower():
                response = None
                seed = random.randint(0,1000)
                temperature = random.random()
        except openai.error.InvalidRequestError as e:
            raise e
        except openai.error.RateLimitError as e:
            if e.error['message'].startswith('Rate limit'):
                time.sleep(3)
            else:
                raise e
        except Exception as e:
            print(e)
            time.sleep(3)
    if response is None:
        raise ValueError("After trying 10 times, the response is still None")
    response.update(config)
    return response

def cache_result(cache):
    if cache['mode']=='generation':
        cur_generation = cache["choices"][0]["message"]["content"]
        cur_generation = re.sub(r'\n+', '\n', cur_generation).strip()
        cur_generation = extract_program(cur_generation,lan=LANGUAGE_MAP[cache['task']])
        cache['generated_code'] = cur_generation
    else:
        try:
            explanations = json.loads(cache["choices"][0]["message"]["content"])
        except:
            with open(os.path.join(cache['problem_dir'],'generated_code')) as f:
                program = f.read()
            program_lines = program.split('\n')
            explanations = []
            for line_id,line in enumerate(program_lines):
                if line_id<30:
                    explanations.append({"syntax": line, "explanation": line, "line ids": [line_id]})
                else:
                    break
        cache['explanations'] = explanations
    with open(os.path.join(cache['problem_dir'], f"{cache['output_tag']}_{cache['mode']}.json"),'w') as f:
        json.dump(cache, f, indent=2)

def gpt_generate(configs,**kwargs):
    num_worker = kwargs['num_worker']
    if num_worker==0:
        num_worker = mp.cpu_count()//2
    count = 0
    if num_worker>1:
        with mp.Pool(num_worker) as pool, tqdm(total=len(configs),desc='GPT') as pbar:
            for result in pool.imap_unordered(gpt_worker, configs):
                pbar.update()
                if result is None:
                    continue
                count += 1
                cache_result(cache=result)
    else:
        for cur_config in configs:
            result = gpt_worker(cur_config)
            if result is None:
                    continue
            count += 1
            cache_result(cache=result)

def codellama_generate(configs,**kwargs):
    batch_size = kwargs['batch_size']
    tokenizer = kwargs['tokenizer']
    pipeline = kwargs['pipeline']
    configs = sorted(configs,key=lambda x:len(tokenizer(x['prompt'])['input_ids']),reverse=True)
    for i in trange(0,len(configs),batch_size,desc='CodeLlama inference'):
        cur_batch = configs[i:i+batch_size]
        batch_exist = True
        for e in cur_batch:
            if not os.path.isfile(os.path.join(e['problem_dir'],f"{e['output_tag']}_generation.json")):
                batch_exist = False
        if not batch_exist:
            cur_prompts = [x['prompt'] for x in cur_batch]
            cur_max_length = max([len(tokenizer(p)['input_ids']) for p in cur_prompts])+400
            assert cur_max_length<8192,f"Including 400 tokens to be generated, the current maximum length is {cur_max_length}"
            try:
                cur_results = pipeline(
                    cur_prompts,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=cur_max_length,
                    batch_size=batch_size
                )
            except Exception as e:
                skip_ids = [iter_example['problem_dir'].split('/')[-1] for iter_example in cur_batch]
                skip_ids_str = ', '.join(skip_ids)
                print(e)
                print(f"{skip_ids_str} are skipped")
                continue
            assert len(cur_results)==len(cur_batch)
            for result1,example1 in zip(cur_results,cur_batch):
                raw_generation = result1[0]['generated_text']
                example1["choices"] = [{"message": {"content": raw_generation}}]
                cur_generation = raw_generation[len(example1['prompt']):]
                cur_generation_components = cur_generation.split('[DONE]')
                cur_generation = cur_generation_components[0]
                cur_generation_lines = cur_generation.split('\n')
                processed_generation_lines = []
                for l in cur_generation_lines:
                    if l.startswith('#') or l.startswith('/') or l=='':
                        continue
                    processed_generation_lines.append(l)
                example1["generated_code"] = '\n'.join(processed_generation_lines)
                with open(os.path.join(example1['problem_dir'],f"{example1['output_tag']}_generation.json"),'w') as f:
                    json.dump(example1,f,indent=2)

def starchat_generate(configs,**kwargs):
    batch_size = kwargs['batch_size']
    tokenizer = kwargs['tokenizer']
    pipeline = kwargs['pipeline']
    configs = sorted(configs,key=lambda x:len(tokenizer(x['prompt'])['input_ids']),reverse=True)
    for i in trange(0,len(configs),batch_size,desc='StarChat inference'):
        cur_batch = configs[i:i+batch_size]
        batch_exist = True
        for e in cur_batch:
            if not os.path.isfile(os.path.join(e['problem_dir'],f"{e['output_tag']}_generation.json")):
                batch_exist = False
        if not batch_exist:
            cur_prompts = [x['prompt'] for x in cur_batch]
            cur_max_length = max([len(tokenizer(p)['input_ids']) for p in cur_prompts])+200
            assert cur_max_length<8192,f"Including 400 tokens to be generated, the current maximum length is {cur_max_length}"
            try:
                cur_results = pipeline(
                    cur_prompts,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=cur_max_length,
                    batch_size=batch_size
                )
            except Exception as e:
                skip_ids = [iter_example['problem_dir'].split('/')[-1] for iter_example in cur_batch]
                skip_ids_str = ', '.join(skip_ids)
                print(e)
                print(f"{skip_ids_str} are skipped")
                continue
            assert len(cur_results)==len(cur_batch)
            for result1,example1 in zip(cur_results,cur_batch):
                raw_generation = result1[0]['generated_text']
                example1["choices"] = [{"message": {"content": raw_generation}}]
                cur_generation = raw_generation[len(example1['prompt']):]
                cur_generation_components = cur_generation.split('```')
                cur_generation = cur_generation_components[0]
                cur_generation_lines = cur_generation.split('\n')
                processed_generation_lines = []
                for l in cur_generation_lines:
                    if l.startswith('#') or l.startswith('/') or l=='':
                        continue
                    processed_generation_lines.append(l)
                example1["generated_code"] = '\n'.join(processed_generation_lines)
                with open(os.path.join(example1['problem_dir'],f"{example1['output_tag']}_generation.json"),'w') as f:
                    json.dump(example1,f,indent=2)
