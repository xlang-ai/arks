import json
import os
import openai
import argparse
import warnings
from arks.method import inference
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    warnings.filterwarnings("ignore")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_tag", type=int, default=0)
    parser.add_argument("--doc_max_length", type=int, default=-1)
    parser.add_argument("--exp_max_length", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--query", type=str, default='',choices=['question','code','code_explanation','execution_feedback'])
    parser.add_argument("--knowledge", type=str, default='',choices=['web_search','documentation','code_snippets','execution_feedback',
                                                                     "documentation_code_snippets","documentation_execution_feedback",
                                                                     "code_snippets_execution_feedback",
                                                                     "documentation_code_snippets_execution_feedback"])
    parser.add_argument("--task", type=str, default="Pony", choices=['Pony','Ring','ScipyM','TensorflowM'])
    parser.add_argument("--openai_org", type=str, default=None)
    parser.add_argument("--openai_key", type=str, default=None)
    parser.add_argument("--prompt_max_length", type=int, default=3696)
    parser.add_argument("--data_dir", type=str, default='arks_data')
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--generator", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--retriever", type=str, default="hkunlp/instructor-large")
    args = parser.parse_args()

    argparse_dict = vars(args)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.output_tag}_args.json"), 'w') as f:
        json.dump(argparse_dict, f, indent=2)
    print('output:',args.output_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir,exist_ok=True)
    if args.openai_org is not None:
        openai.organization = args.openai_org
    if args.openai_key is not None:
        openai.api_key = args.openai_key

    inference(task=args.task,
              query=args.query,
              knowledge=args.knowledge,
              data_dir=args.data_dir,
              output_dir=args.output_dir,
              retriever=args.retriever,
              generator=args.generator,
              output_tag=args.output_tag,
              doc_max_length=args.doc_max_length,
              exp_max_length=args.exp_max_length,
              batch_size=args.batch_size,
              num_worker=args.num_worker)


