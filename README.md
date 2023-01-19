# NLP Final Project - SQL-To-Text


## Project Overview
In this project a pre-trained [T5 model](https://arxiv.org/pdf/1910.10683.pdf), specifically the [CodeT5-base model](https://arxiv.org/pdf/2109.00859.pdf), is fine-tuned on the [WikiSQL dataset](https://github.com/salesforce/WikiSQL) to perform SQL-to-text translations. It is shown that this fine-tuned model achieves a higher [BLEU score](https://aclanthology.org/P02-1040.pdf) than several baseline models. This project also performs human evaluation on the fine-tuned model's predictions, to further assess its viability in performing SQL-to-text translations.
![SQL-to-text demo](https://raw.githubusercontent.com/lewisc4/SQL-To-Text/main/SQL-to-text%20Demo.gif)

## Setting Up The Environment
### Package Installation
It is necessary to have python >= 3.7 installed in order to run the code for this project. In order to install the necessary libraries and modules follow the below instructions.

1. Clone or download this project to your local computer.
2. Navigate to the `code\` directory, where the `setup.py` file is located.
3. Install the `sql_to_text` module and all dependencies by running the following command from the command line: `pip install -e .`
**GPU specific instructions:**
    1. If you are on a GPU machine, you need to install a GPU version of pytorch. To do that, first check what CUDA version your server has with `nvidia-smi`.
    2. If your CUDA version is below 10.2, don't use this server
    3. If your CUDA version is below 11, run `pip install torch`
    4. Else, `pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
    5. Check that pytorch-GPU works via `python -c "import torch; print(torch.cuda.is_available())"`. If it returns False, reinstall pytorch via one of the above command (usually this helps).
    6. If you are using 30XX, A100 or A6000 GPU, you have to use CUDA 11.3 and above. 

## Training A Model
### Hyperparameters
The available hyperparameters for fine-tuning the CodeT5-base model can be found in the `sql_to_text/utils.py` file. By default, a large majority of the hyperparameters are inherited from the CodeT5-base models original parameters. The default model/tokenizer is `Salesforce/codet5-base` (shouldn't be changed) and the default dataset is `wikisql` (shouldn't be changed). However, useful parameters to change/test with are:

* `output_dir` <- Where to save the model to (defaults to `code/cli/Outputs/`)
* `learning_rate` <- The external learning rate
* `batch_size` <- Batch size used by the model
* `weight_decay` <- The external weight decay
* `eval_every_steps` <- How often to evaluate the model (compute BLEU)
* `beam_size` <- Beam size to use during evaluation
* `debug` <- Whether to run in debug mode (uses only 100 examples) or not
* `num_train_epochs` <- Number of training epochs to use
* `train_from_scratch` <- Whether to train the model from scratch or not

### CLI Commands
The below commands can be run from the `cli` directory. By default, the model is saved to the `code/cli/Outputs/` directory. If the provided `output_dir` does not exist, it will automatically be created.

**To train a model that achieves 27+ BLEU scores:**
* `python3 train.py --output_dir=Outputs --num_train_epochs=10 --batch_size=8 --learning_rate=3e-4 --eval_every=10000 --beam_size=10`
* Sample [WandB run](https://wandb.ai/clewis7744/sql_to_text/runs/3exrerr3)

**To train a model that perfectly fits 100 training examples:**
* `python3 train.py --output_dir=Outputs --num_train_epochs=10 --batch_size=8 --learning_rate=1e-3 --debug`
* Sample [WandB run](https://wandb.ai/clewis7744/sql_to_text/runs/2wdqcorw)

**To train a CodeT5-base model from scratch:**
* `python3 train.py --output_dir=Outputs --batch_size=8 --learning_rate=3e-4 --beam_size=10 --max_train_steps=2500 --train_from_scratch`
* Sample [WandB run](https://wandb.ai/clewis7744/sql_to_text/runs/35f2sys4)


## Performing Human Evaluation
### Overview
The `code/cli/human_eval.py` file performs the human evaluations/BLEU score calculations. By default, the `code/cli/Outputs/human_eval_data.csv` file is used, because it contains the input, target, and generated sequences and the corresponding human evaluation score. Running `human_eval.py` computes the BLEU score for these examples and calculates/plots the correlations between the two scores.
### CLI Commands
If `human_eval_data.csv` file does not exist in the `output_dir`, the model generations will be made again on 110 test examples, and will be saved to this file. If the `human_eval_data.csv` file does exist, but has no `Results` column, nothing happens, because there is nothing to do.

**Command to perform human evaluation:**
* `python3 human_eval.py --output_dir=Outputs`

## Running A Streamlit Demo
To run the streamlit demo, the model should be stored in the `code\cli\Outputs` directory.
### CLI Commands
**Command to run Streamlit demo:**
* `streamlit run streamlit_demo.py`
### Demo example
![SQL-to-text demo](https://raw.githubusercontent.com/lewisc4/SQL-To-Text/main/SQL-to-text%20Demo.gif)
