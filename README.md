# Doing Legal Judgment Prediction in the Right Way with Large Language Models

## Installation

You first need to install PyTorch.
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) for more details
specifically for the platforms.

When PyTorch has been installed, you can install requirements from source by cloning the repository and running:
First clone the repository using the anonymouse repository, we will update the cloning script upon acceptance of the
paper.

```bash
cd uksc
pip install -r requirements.txt
```

## Experiment Results

You can easily run open-source llm experiments using following command and altering the parameters as you wish.

```bash
python -m experiments.open_llms --model_name meta-llama/Llama-2-7b-chat-hf --visible_cuda_devices 0,1,2 --run_mode tag
```

In order to use the meta-llama/Meta-Llama-3.1-8B-Instruct model, you have to add the padding token to tokeniser and
resize the embeddings in the model.
You can get this done by executing the following script.

```bash
python -m util.llama3_tokeniser_extend --model_name meta-llama/Meta-Llama-3.1-8B-Instruct
```

This will add the padding token to the tokeniser and will prepare the model in /local_models path for the experiments.
After this action, you can simply use the following script to execute experiments using
meta-llama/Meta-Llama-3.1-8B-Instruct model.

```bash
python -m experiments.llama3 --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --visible_cuda_devices 0,1,2 --run_mode tag
```

In order to execute GPT based experiments first you need to setup environment variable OPENAI_API_KEY with your openai
token, then you can use the following script; where you can manually change the mode of execution and the model you want to experiment with.

```bash
python -m experiments.chatgpt_exp
```

## Parameters

Please find the detailed descriptions of the parameters

```texts
model_name              : Huggingface transformer model name that you need to experiment with; ex: google-bert/bert-base-multilingual-cased
visible_cuda_devices    : Comma separated device numbers to run the experiment on
run_mode                : Either default or tag, the type of experiments you want to run. tag; with legal area, default; without legal area
```

After the execution, the files will be saved on /outputs path. Use these files to manually post edit in classification
decisions and execute following scrpits to collect all the results to evaluation folder.

```bash
python -m data.postprocess.collect_clear_decisions
python -m data.postprocess.collect_clear_reasons
```

There after you can run evaluations using following script,

```bash
python -m evaluation.evaluate
```

In order to evaluate the two approaches, you need to change util/setting.py True/False for legal area tagged and not
tagged approach and execute the above script.

Your results will be in /results path. 
