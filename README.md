# EVOR: Evolving Retrieval for Code Generation

This repository contains the code for our paper [EVOR: Evolving Retrieval for Code Generation](https://arxiv.org/abs/2402.12317). Please refer to our [project page](https://arks-codegen.github.io/) for a quick project overview.

We introduce **EVOR**, a general pipeline for retrieval-augmented code generation (RACG). 
We construct a knowledge soup integrating web search, documentation, execution feedback, and evolved code snippets.
Through ***active retrieval in knowledge soup***, we demonstrate significant increase in benchmEVOR about updated libraries and long-tail programming languages (8.6% to 34.6% in ChatGPT)

## Installation
It is very easy to use EVOR for RACG tasks. In your local machine, we recommend to first create a virtual environment:
```bash
conda env create -n EVOR python=3.8
git clone https://github.com/xlang-ai/EVOR
```
That will create the environment `EVOR` we used. To use the embedding tool, first install the `EVOR` package
```bash
pip install -e .
```
To Evaluate on updated libraries, install the packages via
```bash
cd updated_libraries/ScipyM
pip install -e .
cd ../TensorflowM
pip install -e .
```

### Environment setup

Activate the environment by running
```bash
conda activate EVOR
```

### Data
Please download the [data](https://drive.google.com/file/d/1g_i6Xyl5wFBeXsQGG5kHzCHwtJiHgTq9/view?usp=sharing) and unzip it with password `EVORdata`

You can also access the data in [huggingface](https://huggingface.co/datasets/xlangai/EVOR_data)

load one dataset:
```
from datasets import load_dataset
data_files = {"corpus": "Pony/Pony_docs.jsonl"}
dataset = load_dataset("xlangai/EVOR_data", data_files=data_files)
```

load several datasets:
```
from datasets import load_dataset
data_files = {"corpus": ["Pony/Pony_docs.jsonl", "Ring/Ring_docs.jsonl"]}
dataset = load_dataset("xlangai/EVOR_data", data_files=data_files)
```

## Getting Started
Run inference
```bash
python run.py --output_dir {output_dir} --output_tag {running_flag} --openai_key {your_openai_key} --task {task_name}
```
* --output_tag is the running flag that starts from 0. By simply increasing it, we active the active retrieval process
* --task specifies the task name. We can choose from ScipyM, TensorflowM, Ring or Pony.
* --query specifies the query formulation. Available choices include question, code, code_explanation, execution_feedback.
* --knowledge specifies the knowledge to augment LLM. Available choices include web_search, documentation, code_snippets, execution_feedback, documentation_code_snippets, documentation_execution_feedback, code_snippets_execution_feedback, documentation_code_snippets_execution_feedback
* --doc_max_length specfies the maximum length for documentation
* --exp_max_length specifies the maximum length for code snippets

Run evaluation
```bash
python eval/{task}.py --output_dir {output_dir} --turn {output_flag}
```
This should report the execution accuracy of the inference

