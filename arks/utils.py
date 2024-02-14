import os
import re
import sys
import importlib.util
from collections import defaultdict
from transformers import PreTrainedTokenizerBase

def get_str_len(s, model="gpt-3.5-turbo-1106"):
    if model is None or isinstance(model,str):
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-1106")
        return len(encoding.encode(s))
    elif isinstance(model,PreTrainedTokenizerBase):
        return len(model(s)['input_ids'])
    else:
        raise ValueError(f"The tokenizer model {model} has not been implemented yet")

def extract_program(a_string,lan='python',first_block_only=False):
    indices_object = re.finditer(pattern="```", string=a_string)
    indices = [index.start() for index in indices_object]
    contents = ''
    if len(indices) == 0:
        contents = a_string
    elif len(indices) % 2 == 0:
        for i in range(0, len(indices), 2):
            cur_str = a_string[indices[i]:indices[i + 1]]
            if cur_str.startswith(f"```{lan}"):
                cur_str = cur_str[len(f"```{lan}"):]
            elif cur_str.startswith(f"```\n{lan}"):
                cur_str = cur_str[len(f"```\n{lan}"):]
            elif cur_str.startswith("```"):
                cur_str = cur_str[len("```"):]
            contents += cur_str
            if first_block_only:
                break
    else:
        if lan=='ring':
            contents = a_string.replace(f"```{lan}", '').replace("```", '')
        else:
            contents = a_string.replace(f"```{lan}", '').replace("```", '').replace(f"{lan}\n", '')
    lines = contents.strip().split('\n')
    if lines[-1].isidentifier():
        contents = '\n'.join(lines[:-1])
    if lan=='ring':
        return contents
    return contents.replace(f"{lan}\n", '')

def clean_str(s):
    return s.encode('utf-8', errors='ignore').decode('utf-8')

def check_dir_txt(d):
    if not os.path.isdir(d):
        return False
    for file in os.listdir(d):
        if file.endswith('.txt'):
            return True
    return False

def find_error_lines_python(problem_dir):
    error_lines = defaultdict(list)
    error_messages = {}
    if not check_dir_txt(os.path.join(problem_dir,'logs')):
        print('no txt file found in',os.path.join(problem_dir,'logs'))
        exit(0)
    for log_file in os.listdir(os.path.join(problem_dir,'logs')):
        if not log_file.endswith('.txt'):
            continue
        with open(os.path.join(problem_dir,'logs',log_file)) as err_log:
            trackback = err_log.read()
        trackback_lines = trackback.strip().split('\n')
        error_messages[log_file] = [trackback_lines[-1].strip()]
        error_idx = len(trackback_lines) - 1
        while error_idx >= 0:
            if 'program.py' in trackback_lines[error_idx]:
                assert error_idx < len(trackback_lines) - 1
                line_component = trackback_lines[error_idx].split(',')[1].strip()
                assert line_component.startswith('line'),f"{trackback_lines[error_idx].split(',')}\n{os.path.join(problem_dir,'logs',log_file)}"
                line_components = line_component.split(' ')
                assert len(line_components)==2
                line_idx = int(line_components[1].strip())
                error_lines[log_file].append((line_idx,trackback_lines[error_idx + 1].strip()))
            if trackback_lines[error_idx].startswith('Traceback '):
                break
            error_idx -= 1
    error_lines_keys = list(error_lines.keys())
    for i in range(len(error_lines_keys)):
        for j in range(i):
            if error_lines_keys[i] in error_lines and \
                error_lines[error_lines_keys[i]]==error_lines[error_lines_keys[j]] and \
                error_messages[error_lines_keys[i]]==error_messages[error_lines_keys[j]]:
                error_lines.pop(error_lines_keys[i])
                error_messages.pop(error_lines_keys[i])
    return error_lines,error_messages

def find_error_lines_ring(problem_dir: str) -> (dict,dict):
    error_lines = defaultdict(list)
    error_messages = defaultdict(list)
    if not check_dir_txt(os.path.join(problem_dir,'logs')):
        print('no txt file found in',os.path.join(problem_dir,'logs'))
        exit(0)
    if os.path.isdir(os.path.join(problem_dir,'logs')):
        for log_file in os.listdir(os.path.join(problem_dir,'logs')):
            if not log_file.endswith('.txt'):
                continue
            with open(os.path.join(problem_dir,'logs',log_file)) as err_log:
                trackback = err_log.read()
            with open(os.path.join(problem_dir,'program.ring')) as program_file:
                cur_program_lines = program_file.readlines()
            num_program_lines = len(cur_program_lines)
            trackback_lines = trackback.strip().split('\n')
            for cur_traceback_line in trackback_lines:
                for error_idx in range(num_program_lines):
                    if f"Line ({error_idx+1}) Error" in cur_traceback_line or f"Line {error_idx+1} Error" in cur_traceback_line:
                        error_messages[log_file].append(':'.join(cur_traceback_line.split(':')[1:]).strip())
                    if cur_traceback_line.startswith('Line'):
                        error_messages[log_file].append(cur_traceback_line)
                    error_lines[log_file].append((error_idx + 1, cur_program_lines[error_idx].strip()))
    error_lines_keys = list(error_lines.keys())
    for i in range(len(error_lines_keys)):
        for j in range(i):
            if error_lines_keys[i] in error_lines and \
                error_lines[error_lines_keys[i]]==error_lines[error_lines_keys[j]] and \
                error_messages[error_lines_keys[i]]==error_messages[error_lines_keys[j]]:
                error_lines.pop(error_lines_keys[i])
                error_messages.pop(error_lines_keys[i])
    return error_lines,error_messages

def find_error_lines_pony(problem_dir: str) -> (dict,dict):
    error_lines = defaultdict(list)
    error_messages = defaultdict(list)
    if not check_dir_txt(os.path.join(problem_dir,'logs')):
        print('no txt file found in',os.path.join(problem_dir,'logs'))
        exit(0)
    for log_file in os.listdir(os.path.join(problem_dir,'logs')):
        if not log_file.endswith('.txt'):
            continue
        with open(os.path.join(problem_dir,'logs',log_file)) as err_log:
            trackback = err_log.read()
        trackback = re.sub(r'\n+', '\n', trackback).strip()
        with open(os.path.join(problem_dir,'program','main.pony')) as program_file:
            cur_program = program_file.read()
        cur_program_lines = cur_program.split('\n')
        num_program_lines = len(cur_program_lines)
        trackback_lines = trackback.strip().split('\n')
        for traceback_idx,cur_traceback_line in enumerate(trackback_lines):
            for error_idx in range(num_program_lines):
                if f"program/main.pony:{error_idx+1}:" in cur_traceback_line:
                    error_messages[log_file].append(cur_traceback_line.split(':')[-1].strip())
                    error_lines[log_file].append((error_idx+1, cur_program_lines[error_idx].strip()))
                    assert cur_program_lines[error_idx].strip()==trackback_lines[traceback_idx+1].strip() or \
                        cur_program_lines[error_idx].strip().startswith(trackback_lines[traceback_idx+1].strip()),\
                        f"{cur_program}\n-------\n{trackback}\n-------\n" \
                        f"{cur_program_lines[error_idx].strip()}\n-------\n{trackback_lines[traceback_idx+1].strip()}\n"
    error_lines_keys = list(error_lines.keys())
    for i in range(len(error_lines_keys)):
        for j in range(i):
            if error_lines_keys[i] in error_lines and \
                error_lines[error_lines_keys[i]]==error_lines[error_lines_keys[j]] and \
                error_messages[error_lines_keys[i]]==error_messages[error_lines_keys[j]]:
                error_lines.pop(error_lines_keys[i])
                error_messages.pop(error_lines_keys[i])
    return error_lines,error_messages

def import_source_file(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, fname)
    if spec is None:
        raise ImportError(f"Could not load spec for module '{modname}' at: {fname}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError as e:
        raise ImportError(f"{e.strerror}: {fname}") from e
    return module