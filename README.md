# SQL-To-Text Translations Using Deep Learning & NLP

## Project Overview

In this project, a pre-trained [T5 model](https://arxiv.org/pdf/1910.10683.pdf), specifically the [CodeT5-base model](https://arxiv.org/pdf/2109.00859.pdf), was fine-tuned on the [WikiSQL dataset](https://github.com/salesforce/WikiSQL) to perform SQL-to-text translations. This fine-tuned model achieved a higher [BLEU score](https://aclanthology.org/P02-1040.pdf) than several baseline models. This project also performed human evaluation on the fine-tuned model's predictions, to further assess its viability in performing SQL-to-text translations.

![SQL-to-text demo](https://raw.githubusercontent.com/lewisc4/SQL-To-Text/main/SQL-to-text%20Demo.gif)


## Environment Setup

### Package Installation
It is necessary to have python >= 3.7 installed in order to run the code for this project. In order to install the necessary libraries and modules follow the below instructions.

1. Clone or download this project to your local computer.
2. Navigate to the [root directory](https://github.com/lewisc4/SQL-To-Text), where the [`setup.py`](/setup.py) file is located.
3. Install the [`sql_to_text`](/sql_to_text) module and all dependencies by running the following command: `pip install -e .` (required python modules are in [`requirements.txt`](/requirements.txt)).

### GPU-related Requirements/Installations
Follow the steps below to ensure your GPU and all relevant libraries are up to date and in good standing.

1. If you are on a GPU machine, you need to install a GPU version of pytorch. To do that, first check what CUDA version your server has with `nvidia-smi`.
2. If your CUDA version is below 10.2, don't use this server
3. If your CUDA version is below 11, run `pip install torch`
4. Else, `pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
5. Check that pytorch-GPU works via `python -c "import torch; print(torch.cuda.is_available())"`. If it returns False, reinstall pytorch via one of the above command (usually this helps).
6. If you are using 30XX, A100 or A6000 GPU, you have to use CUDA 11.3 and above.


## Training
The [`train.py`](/cli/train.py) script is used to train a model and save it to a specified directory, which is created if it doesn't exist.

### Hyperparameters
The available hyperparameters for fine-tuning the CodeT5-base model can be found in [`utils.py`](/sql_to_text/utils.py). By default, a large majority of the hyperparameters are inherited from the CodeT5-base model's original parameters. The default model/tokenizer is `Salesforce/codet5-base` and the default dataset is `wikisql` (both shouldn't be changed). However, useful parameters to change/test with are:

* `output_dir` <- Where to save the model (created if it doesn't exist, defaults to [`cli/Outputs/`](/cli/Outputs))
* `learning_rate` <- The external learning rate
* `batch_size` <- Batch size used by the model
* `weight_decay` <- The external weight decay
* `eval_every_steps` <- How often to evaluate the model (compute BLEU)
* `beam_size` <- Beam size to use during evaluation
* `debug` <- Whether to run in debug mode (uses only 100 examples) or not
* `num_train_epochs` <- Number of training epochs to use
* `train_from_scratch` <- Whether to train the model from scratch or not

### Example Usage
**To train a model that achieves a 27+ BLEU score (sample [WandB run](https://wandb.ai/clewis7744/sql_to_text/runs/3exrerr3)):**
- `python3 train.py --output_dir=Outputs --num_train_epochs=10 --batch_size=8 --learning_rate=3e-4 --eval_every=10000 --beam_size=10`

**To train a model that perfectly fits 100 training examples (sample [WandB run](https://wandb.ai/clewis7744/sql_to_text/runs/2wdqcorw)):**
- `python3 train.py --output_dir=Outputs --num_train_epochs=10 --batch_size=8 --learning_rate=1e-3 --debug`

**To train a CodeT5-base model from scratch (sample [WandB run](https://wandb.ai/clewis7744/sql_to_text/runs/35f2sys4)):**
- `python3 train.py --output_dir=Outputs --batch_size=8 --learning_rate=3e-4 --beam_size=10 --max_train_steps=2500 --train_from_scratch`


## Human Evaluation

The [`human_eval.py`](/cli/human_eval.py) script is used to compare BLEU scores with human evaluation scores corresponding to the same model-generated sequences. By default, [`human_eval_data.csv`](/cli/Outputs/human_eval_data.csv) is used as input data for human evaluations. In this file, each datapoint (i.e. row) consists of a human evaluation score and its corresponding input, target, and model-generated sequence.

If [`human_eval_data.csv`](/cli/Outputs/human_eval_data.csv) does not exist (in `output_dir`), model generations will be made using 110 test set examples to create it. However, it must also have a `Results` column with human evaluation scores for each example, otherwise nothing will happen.

### Example Usage
**To run [`human_eval.py`](/cli/human_eval.py):**
- `python3 human_eval.py --output_dir=Outputs`


## Streamlit Demo

[`streamlit_demo.py`](/cli/streamlit_demo.py)is the script used to initialize and drive an interactive Streamlit demo, using a saved SQL-to-text model. For the demo to work, the desired model to demo must be saved in [`cli/Outputs/`](/cli/Outputs). An example demo can be found under **[Project Overview](https://github.com/lewisc4/SQL-To-Text/blob/main/README.md#project-overview)**. 

### Example Usage
**To run [`streamlit_demo.py`](/cli/streamlit_demo.py):**
- `streamlit run streamlit_demo.py`

