# ARKS: Active Retrieval in Knowledge Soup for Code Generation

This repository contains the code for our paper [ARKS: Active Retrieval in Knowledge Soup for Code Generation](https://github.com/). Please refer to our [project page](https://github.com/) for a quick project overview.

We introduce **ARKS**, a general pipeline for retrieval-augmented code generation (RACG). 
We construct a knowledge soup integrating web search, documentation, execution feedback, and evolved code snippets.
Through ***active retrieval in knowledge soup***, we demonstrate significant increase in benchmarks about updated libraries and long-tail programming languages (8.6% to 34.6% in ChatGPT)

## Installation
It is very easy to use ARKS for RACG tasks. In your local machine, we recommend to first create a virtual environment:
```bash
conda env create -n arks python=3.8
git clone https://github.com/hongjin-su/arks_dev
```
That will create the environment `arks` we used. To use the embedding tool, first install the `arks` package
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
conda activate arks
```

### Data
Please download the [data](https://drive.google.com/file/d/1g_i6Xyl5wFBeXsQGG5kHzCHwtJiHgTq9/view?usp=sharing) and unzip it with password `arksdata`

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

