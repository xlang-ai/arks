import os
import torch
import json
import shutil
from arks.data import ArksData
from InstructorEmbedding import INSTRUCTOR
from transformers import AutoTokenizer
from transformers import pipeline as transformer_pipeline
from arks.model import codellama_generate,gpt_generate,starchat_generate
from arks.retrieval import formulate_query,retrieve_knowledge
from arks.utils import find_error_lines_pony,find_error_lines_ring,find_error_lines_python,get_str_len

FIND_ERROR_LINE = {
    "TensorflowM": find_error_lines_python,
    "ScipyM": find_error_lines_python,
    'Ring': find_error_lines_ring,
    'Pony': find_error_lines_pony,
}
QUERY_INST = {
    "question": "Represent a coding problem description: ",
    "code": "Represent the code: ",
    'code_explanation': "Represent the code explanation for retrieval: ",
    'execution_feedback': "Represent the error message: ",
}

def get_interpreter_error_prompt(problem_dir,find_error_lines_func):
    with open(os.path.join(problem_dir, 'generated_code')) as f:
        generated_code = f.read()
    error_lines, error_messages = find_error_lines_func(problem_dir=problem_dir)
    generated_error_lines = []
    reported_error_msg = ''
    for key, msg in error_messages.items():
        if key in error_lines:
            lines = error_lines[key]
            for l in lines:
                assert isinstance(l, tuple) and len(l) == 2
                if l[1].strip() in generated_code and not l[1].strip() in generated_error_lines:
                    generated_error_lines.append(l[1].strip())
                    break
        if isinstance(error_messages[key],str):
            reported_error_msg = error_messages[key]
        elif isinstance(error_messages[key],list):
            reported_error_msg = error_messages[key][0]
        else:
            raise ValueError(f"error messages have type {type(error_messages)}")
        if len(generated_error_lines) > 0:
            break
    if len(generated_error_lines) > 0:
        generated_error_lines_str = '\n'.join(generated_error_lines)
        error_prompt = f"There are errors in\n" \
                       f"{generated_error_lines_str}\n\n"
    else:
        error_prompt = ''
    if not 'error' in reported_error_msg.lower():
        error_prompt += f"Error: {reported_error_msg.strip()}\n\n"
    else:
        error_prompt += f"{reported_error_msg.strip()}\n\n"
    return error_prompt

def format_doc_str(cur_docs,max_length,base_prompt,tokenizer=None):
    cur_doc_str = ""
    tmp_prompt = base_prompt
    for document in cur_docs:
        assert isinstance(document, str) and len(document) > 0,f"docstr: {document}"
        tmp_prompt += document
        if get_str_len(s=tmp_prompt,model=tokenizer) < max_length:
            cur_doc_str += f"{document}\n\n"
        else:
            break
    return cur_doc_str

def format_example_str(examples,output_dir,exp_max_length,tokenizer=None):
    exp_str = f"## Examples:\n"
    for code_idx, correct_code_dir in enumerate(examples):
        with open(os.path.join(output_dir, correct_code_dir, 'generated_code')) as code_file:
            cur_correct_code = code_file.read()
        cur_code_str = ''
        code_lines = cur_correct_code.split('\n')
        for code_line_idx, line_iter in enumerate(code_lines):
            if code_line_idx == 0:
                cur_code_str += f">>> {line_iter}\n"
            else:
                cur_code_str += f"... {line_iter}\n"
        cur_code_str += '\n'
        if get_str_len(exp_str + cur_code_str) < exp_max_length:
            exp_str += cur_code_str
        else:
            break
    return exp_str

def inference(task,query,knowledge,data_dir,output_dir,retriever,generator,output_tag,doc_max_length,exp_max_length,batch_size,num_worker):
    if 'instructor' in retriever:
        retrieval_tokenizer = INSTRUCTOR(retriever)
    else:
        raise ValueError(f"Retriever {retriever} is not supported yet")
    data = ArksData(task,data_dir,knowledge,retrieval_tokenizer)
    data_dir = os.path.join(data_dir,task,'examples')
    print(data_dir)
    for sub in os.listdir(data_dir):
        if not sub.startswith('q'):
            continue
        if not os.path.isdir(os.path.join(output_dir,sub)):
            os.makedirs(os.path.join(output_dir,sub),exist_ok=True)
        for subsub in os.listdir(os.path.join(data_dir,sub)):
            if os.path.isdir(os.path.join(data_dir,sub,subsub)) and not os.path.isdir(os.path.join(output_dir,sub,subsub)):
                shutil.copytree(os.path.join(data_dir,sub,subsub),os.path.join(output_dir,sub,subsub))
            elif os.path.isfile(os.path.join(data_dir,sub,subsub)) and not os.path.isfile(os.path.join(output_dir,sub,subsub)):
                shutil.copyfile(os.path.join(data_dir,sub,subsub),os.path.join(output_dir,sub,subsub))
    if generator=='codellama':
        tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-34b-Instruct-hf', model_max_length=4096)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif generator=='starchat':
        tokenizer = AutoTokenizer.from_pretrained('HuggingFaceH4/starchat-beta', model_max_length=4096)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer = None
    if not os.path.isfile(os.path.join(output_dir, 'syntax_correct_ids.json')):
        syntax_correct_ids = []
    else:
        with open(os.path.join(output_dir, 'syntax_correct_ids.json')) as f:
            syntax_correct_ids = json.load(f)
    all_configs = []
    if query=='':
        for subdir in os.listdir(output_dir):
            if not os.path.isdir(os.path.join(output_dir, subdir)) or not subdir.startswith('q'):
                continue
            if subdir in syntax_correct_ids:
                continue
            if os.path.isfile(os.path.join(output_dir,subdir,f"{output_tag}_generation_cache.json")):
                continue
            with open(os.path.join(output_dir, subdir, 'example.json')) as f:
                example = json.load(f)
            if 'execution_feedback' in knowledge:
                with open(os.path.join(output_dir, subdir, 'generated_code')) as f:
                    generated_code = f.read()
                error_prompt = get_interpreter_error_prompt(problem_dir=os.path.join(output_dir, subdir),
                                                        find_error_lines_func=FIND_ERROR_LINE[task])
                error_info = f"\nHere is a wrong implementation:\n{generated_code}\n{error_prompt}Please correct the code."
            else:
                error_info = ''
            if 'gpt' in generator:
                messages = [{"role": "system", "content": "Your are a good programmer."},
                            {"role": "user", "content": example["problem_description"]+f'{error_info}\n\n'+example['instructions']['gpt']}]
                all_configs.append({
                    'messages': messages,
                    'problem_dir': os.path.join(output_dir,subdir),
                    'output_tag': output_tag,
                    'mode': 'generation',
                    'temperature': 0,
                    'model': generator,
                    'task': task
                })
            elif generator=='codellama':
                prompt = '[INST]\n'+example["problem_description"]
                prompt += f"{error_info}\n{example['instructions']['codellama']}"
                all_configs.append({
                    'prompt': prompt,
                    'problem_dir': os.path.join(output_dir,subdir),
                    'output_tag': output_tag
                })
            elif generator=='starchat':
                prompt = '<|system|>\nYou are a good programmer.\n<|end|>\n<|user|>\n'+example["problem_description"]
                prompt += f"{error_info}\n{example['instructions']['starchat']}"
                all_configs.append({
                    'prompt': prompt,
                    'problem_dir': os.path.join(output_dir,subdir),
                    'output_tag': output_tag
                })
            else:
                raise ValueError(f"The generator {generator} is not supported yet")
    else:
        queries = formulate_query(formulation=query,output_dir=output_dir,
                                  find_error_lines_func=FIND_ERROR_LINE[task],
                                  num_worker=num_worker,output_tag=output_tag)
        if not 'doc' in knowledge:
            knowledge_inst = "Represent the code: "
        else:
            knowledge_inst = "Represent the code documentation for retrieval: "
        if 'instructor' in retriever:
            retrieval_model = INSTRUCTOR(retriever)
        else:
            raise ValueError(f"Retriever {retriever} is not supported yet")
        retrieved_docs,retrieved_examples = retrieve_knowledge(data=data,query_inst=QUERY_INST[query],knowledge_inst=knowledge_inst,
                                            retrieval_model=retrieval_model,output_dir=output_dir,queries=queries,
                                            knowledge=knowledge,output_tag=output_tag,task=task)
        for subdir in os.listdir(output_dir):
            if not os.path.isdir(os.path.join(output_dir, subdir)) or not subdir.startswith('q'):
                continue
            if subdir in syntax_correct_ids:
                continue
            if os.path.isfile(os.path.join(output_dir,subdir,f"{output_tag}_generation_cache.json")):
                continue
            with open(os.path.join(output_dir, subdir, 'example.json')) as f:
                example = json.load(f)
            if os.path.isfile(os.path.join(output_dir, subdir, 'generated_code')):
                with open(os.path.join(output_dir, subdir, 'generated_code')) as f:
                    generated_code = f.read()
            else:
                generated_code = ''
            cur_doc_str = format_doc_str(
                cur_docs=retrieved_docs[subdir],
                max_length=doc_max_length,
                base_prompt=example['problem_description']+example['instructions'][generator.split('-')[0]],
                tokenizer=tokenizer
            )
            if "code_snippets" in knowledge:
                exp_str = format_example_str(
                    examples=retrieved_examples[subdir],
                    output_dir=output_dir,
                    exp_max_length=exp_max_length
                )
            else:
                exp_str = ''
            if 'gpt' in generator:
                prompt = cur_doc_str+exp_str+example["problem_description"]
                messages = [{"role": "system", "content": "Your are a good programmer."},
                            {"role": "user", "content": prompt+'\n\n'+example['instructions']['gpt']}]
                if 'execution_feedback' in knowledge:
                    error_prompt = get_interpreter_error_prompt(problem_dir=os.path.join(output_dir, subdir),
                                                            find_error_lines_func=FIND_ERROR_LINE[task])
                    messages += [{"role": "assistant", "content": f"```\n{generated_code}\n```"},
                                 {"role": "user", "content": f"{error_prompt}Please correct the code."}]
                all_configs.append({
                    'messages': messages,
                    'problem_dir': os.path.join(output_dir,subdir),
                    'output_tag': output_tag,
                    'mode': 'generation',
                    'temperature': 0,
                    'model': generator,
                    'task': task
                })
            elif generator=='codellama':
                prompt = '[INST]\n'+cur_doc_str+exp_str+example["problem_description"]
                if 'execution_feedback' in knowledge:
                    error_prompt = get_interpreter_error_prompt(problem_dir=os.path.join(output_dir, subdir),
                                                            find_error_lines_func=FIND_ERROR_LINE[task])
                    prompt += f"Here is a wrong implementation:\n{generated_code}\n{error_prompt}Please correct the code."
                prompt += f"\n{example['instructions']['codellama']}"
                all_configs.append({
                    'prompt': prompt,
                    'problem_dir': os.path.join(output_dir,subdir),
                    'output_tag': output_tag
                })
            elif generator=='starchat':
                prompt = '<|system|>\nYou are a good programmer.\n<|end|>\n<|user|>\n' + cur_doc_str + example["problem_description"]
                if 'execution_feedback' in knowledge:
                    error_prompt = get_interpreter_error_prompt(problem_dir=os.path.join(output_dir, subdir),
                                                            find_error_lines_func=FIND_ERROR_LINE[task])
                    prompt += f"Here is a wrong implementation:\n{generated_code}\n{error_prompt}Please correct the code."
                prompt += f"\n{example['instructions']['starchat']}"
                all_configs.append({
                    'prompt': prompt,
                    'problem_dir': os.path.join(output_dir, subdir),
                    'output_tag': output_tag
                })
            else:
                raise ValueError(f"The generator {generator} is not supported yet")
    if generator=='codellama':
        cur_pipeline = transformer_pipeline(
            "text-generation",
            model='codellama/CodeLlama-34b-Instruct-hf',
            torch_dtype=torch.float16,
            device_map="auto",
        )
        cur_pipeline.tokenizer.pad_token_id = cur_pipeline.tokenizer.eos_token_id
        codellama_generate(
            configs=all_configs,
            batch_size=batch_size,
            tokenizer=tokenizer,
            pipeline=cur_pipeline
        )
    elif generator=='starchat':
        cur_pipeline = transformer_pipeline(
            "text-generation",
            model='HuggingFaceH4/starchat-beta',
            torch_dtype=torch.float16,
            device_map="auto",
        )
        cur_pipeline.tokenizer.pad_token_id = cur_pipeline.tokenizer.eos_token_id
        starchat_generate(
            configs=all_configs,
            batch_size=batch_size,
            tokenizer=tokenizer,
            pipeline=cur_pipeline
        )
    elif 'gpt' in generator:
        gpt_generate(
            configs=all_configs,
            num_worker=num_worker
        )
    else:
        raise ValueError(f"The generator {generator} is not supported yet")




