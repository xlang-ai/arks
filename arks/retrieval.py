import os
import json
import time
from googlesearch import search
from arks.explain import find_error_code_explanations,find_code_explanations,explain
from sklearn.metrics.pairwise import cosine_similarity
from arks.model import LANGUAGE_MAP
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict


SCIPY_REPLACE = {
    'ks_2samp': 'two_sample_ks',
    'cdist': 'dist_mat',
    'ks_1samp': 'one_sample_ks',
    'linear_sum_assignment': 'best_2d_assign',
    'label': 'detect_features',
    'sum_labels': 'group_sum',
    'block_diag': 'block_mat_with_diag',
    'dct': 'discretre_cos_trans',
    'zoom': 'resize',
    'fsolve': 'root_digger',
    'argrelextrema': 'relative_argextreme',
    'cKDTree': 'knn',
    'ndimage': 'rt_array',
    'zscore': 'normalize',
    'minmax': 'normalize',
    'median_filter': 'medianFilter',
    'line_search': 'lineSearch',
    'ranksums': 'wilcoxon_rktest',
    'kurtosis': 'pearson_kurtosis',
    'norm.cdf': 'normal_cdf',
    'norm.ppf': 'normal_ppf',
    'lognorm.cdf': 'lognormal_cdf',
    'uniform.cdf': 'uniform_cdf',
    'binom': 'binom_pmf',
    'interpolate': 'linear_interpolate',
    'curve_fit': 'nonlinear_fit',
    'quad': 'definite_integrate',
    'minimize': 'optimize',
    'maximize': 'optimize',
    'fminbound': 'scalar_minimum',
    'solve_ivp': 'ode_solver',
    'trapz': 'numeric_integrate',
    'griddata': 'NDLinearInterpolator',
    'interp1d': 'NDLinearInterpolator',
    'UnivariateSpline': 'NDSplineInterpolator',
    'RectBivariateSpline': 'NDSplineInterpolator',
    'sparse': 'SparseMatrix',
}

TENSORFLOW_REPLACE = {
    'assign': 'assn_value',
    'numpy': 'numpy_value',
    'reverse': 'rev_tensor',
    'reshape': 'transform_shape',
    'matmul': 'mm',
    'TensorVariable': 'init_variable',
    'one_hot': 'create_one_hot',
    'Tensor': 'init_tensor',
    'constant': 'const',
    'sequence_mask': 'seq_padding_mask',
    'tile': 'repeated_copy',
    'stack': 'pile',
    'concat': 'splice',
    'squeeze': 'delete_axis',
    'expand_dims': 'insert_new_axis',
    'where': 'condition_filling',
    'gather_nd': 'cull_nd',
    'as_str': 'byte2text',
    'as_bytes': 'text2byte',
    'einsum': 'einsum',
    'from_tensor_slices': 'create_dataset_from_tensor',
    'map': 'map',
    'flat_map': 'map_to_flat',
    'argmax': 'arg',
    'argmin': 'arg',
    'set_seed': 'deter_seed',
    'reduce_sum': 'get_summation',
    'reduce_prod': 'get_multiplication',
    'reciprocal': 'multiplicative_inverse',
    'subtract': 'get_subtraction',
    'get_mean': 'get_mean',
    'get_std': 'get_std',
    'ones_like': 'ones_like',
    'ones': 'ones',
    'zeros_like': 'zeros_like',
    'zeros': 'zeros',
    'cast': 'change_dtype'
}

def get_website_contents(url):
    import requests
    from bs4 import BeautifulSoup
    import html2text
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    h = html2text.HTML2Text()
    markdown = h.handle(soup.prettify())
    lines = markdown.split('\n')
    new_lines = []
    for l in lines:
        if l.strip() != '' and l.strip()[0].isalpha():
            new_lines.append(l)
    processed_result = '\n'.join(new_lines[:100])
    return processed_result

def google_search_worker(cur_config):
    num_docs = 2
    results = []
    try:
        for j in search(cur_config['query'], tld="co.in", num=num_docs, stop=num_docs, pause=1):
            try:
                search_contents = get_website_contents(j)
                assert isinstance(search_contents,str)
                if cur_config['task']=='ScipyM':
                    for k,v in SCIPY_REPLACE.items():
                        search_contents = search_contents.replace(k,v)
                elif cur_config['task']=='TensorflowM':
                    for k,v in TENSORFLOW_REPLACE.items():
                        search_contents = search_contents.replace(k,v)
                results.append(search_contents)
            except:
                pass
    except Exception as e:
        print(e)
    return {
        'results': results,
        'problem_dir': cur_config['problem_dir']
    }

def calculate_st(model_st,queries=None,correct=None,query_emb=None,correct_emb=None,error_topk=1,correct_topk=1):
    if query_emb is None:
        query_emb = model_st.encode(queries)
    if correct_emb is None:
        correct_emb = model_st.encode(correct)
    scores = cosine_similarity(correct_emb,query_emb)
    agg_scores = []
    for score in scores:
        agg_scores.append(sum(sorted(score)[-error_topk:]))
    return sum(sorted(agg_scores)[-correct_topk:])

def formulate_query(formulation,output_dir,find_error_lines_func,output_tag,num_worker):
    if not os.path.isfile(os.path.join(output_dir, 'syntax_correct_ids.json')):
        syntax_correct_ids = []
    else:
        with open(os.path.join(output_dir, 'syntax_correct_ids.json')) as f:
            syntax_correct_ids = json.load(f)
    queries = {}
    if formulation=="code_explanation":
        explain(output_dir,output_tag,num_worker)
    for problem_dir in os.listdir(output_dir):
        if not os.path.isdir(os.path.join(output_dir, problem_dir)):
            continue
        if problem_dir in syntax_correct_ids:
            continue
        if formulation=="code_explanation":
            queries[problem_dir] = find_error_code_explanations(problem_dir=os.path.join(output_dir, problem_dir),
                                                                                find_error_lines_func=find_error_lines_func)
        elif formulation=='question':
            with open(os.path.join(output_dir, problem_dir, "example.json")) as f:
                example = json.load(f)
            queries[problem_dir] = [example["problem_description"]]
        elif formulation=='code':
            with open(os.path.join(output_dir, problem_dir, "generated_code")) as f:
                cur_code = json.load(f)
            queries[problem_dir] = [cur_code]
        elif formulation=='execution_feedback':
            _,error_messages = find_error_lines_func(problem_dir=os.path.join(output_dir, problem_dir))
            msg_inst = []
            for _,m_list in error_messages.items():
                assert isinstance(m_list,list)
                for m in m_list:
                    assert isinstance(m,str)
                    msg_inst.append(m)
            queries[problem_dir] = msg_inst
        else:
            raise ValueError(f"{formulation} for query formulation has not been implemented yet!")
    return queries

def retrieve_knowledge(data,query_inst,knowledge_inst,retrieval_model,output_dir,queries,knowledge,output_tag,task):
    if not os.path.isfile(os.path.join(output_dir, 'syntax_correct_ids.json')):
        syntax_correct_ids = []
    else:
        with open(os.path.join(output_dir, 'syntax_correct_ids.json')) as f:
            syntax_correct_ids = json.load(f)
    retrieved_exps = {}
    if "code_snippets" in knowledge or "documentation" in knowledge:
        docs = data.get_docs()
        doc_emb_cache = {}
        idx_map = {}
        total = []
        for func, documents in docs.items():
            documents_inst = [[knowledge_inst, d] for d in documents]
            assert len(documents_inst)!=0
            start = len(total)
            total += documents_inst
            end = len(total)
            idx_map[func] = [start,end]
        total_emb = retrieval_model.encode(total,batch_size=128,show_progress_bar=True)
        for func,indices in idx_map.items():
            doc_emb_cache[func] = total_emb[indices[0]:indices[1]]
            assert len(doc_emb_cache[func])>0,f"{indices}"
        retrieved_docs = {}
        correct_code_explanation_emb = {}
        if "code_snippets" in knowledge:
            for problem_dir in os.listdir(output_dir):
                if not problem_dir.startswith('q') or not os.path.isdir(os.path.join(output_dir,problem_dir)) or \
                        not problem_dir in syntax_correct_ids:
                    continue
                cur_code_explanations = find_code_explanations(os.path.join(output_dir,problem_dir))
                cur_code_explanations_inst = [["Represent the code explanation for retrieval: ",code_exp] for code_exp in cur_code_explanations]
                correct_code_explanation_emb[problem_dir] = retrieval_model.encode(cur_code_explanations_inst)
        for problem_dir in os.listdir(output_dir):
            if not os.path.isdir(os.path.join(output_dir, problem_dir)):
                continue
            if problem_dir in syntax_correct_ids:
                continue
            queries_inst = [[query_inst, e] for e in queries[problem_dir]]
            queries_emb = retrieval_model.encode(queries_inst)
            doc_scores = {}
            for func, doc_emb in doc_emb_cache.items():
                cur_doc_score = calculate_st(model_st=retrieval_model,
                                            query_emb=queries_emb,
                                            correct_emb=doc_emb)
                doc_scores[func] = cur_doc_score
            doc_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            selected_docs = []
            for item in doc_scores:
                if data.doc_split_token.join(docs[item[0]])=='':
                    print('doc empty',item[0])
                    exit(0)
                selected_docs.append(data.doc_split_token.join(docs[item[0]]))
            retrieved_docs[problem_dir] = selected_docs

            if "code_snippets" in knowledge:
                exp_scores = {}
                for correct_code_dir, exp_emb in correct_code_explanation_emb.items():
                    cur_exp_score = calculate_st(model_st=retrieval_model,
                                                query_emb=queries_emb,
                                                correct_emb=exp_emb)
                    exp_scores[correct_code_dir] = cur_exp_score
                exp_scores_sort = sorted(exp_scores.items(),key=lambda x:x[1],reverse=True)
                retrieved_exps[problem_dir] = [dir_score[0] for dir_score in exp_scores_sort]
    elif 'web_search' in knowledge:
        search_configs = []
        for question_dir in os.listdir(output_dir):
            if not os.path.isdir(os.path.join(output_dir, question_dir)) or \
                not question_dir.startswith('q') or question_dir in syntax_correct_ids or \
                    os.path.isfile(os.path.join(output_dir, question_dir,f"{output_tag}_generation.json")):
                continue
            for q in queries[question_dir][:2]:
                search_configs.append({
                    'query': q + ' ' + LANGUAGE_MAP[task],
                    'problem_dir': os.path.join(output_dir, question_dir),
                    'output_tag': output_tag,
                    'task': task
                })
        retrieved_docs = defaultdict(list)
        with mp.Pool(1) as pool, tqdm(total=len(search_configs), desc='google search') as pbar:
            for return_contents in pool.imap_unordered(google_search_worker, search_configs):
                pbar.update()
                retrieved_docs[return_contents['problem_dir']] += return_contents['results']
    else:
        raise ValueError(f"Knowledge {knowledge} is not supported yet")
    return retrieved_docs,retrieved_exps

