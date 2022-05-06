import random
import argparse
import torch
import transformers
import numpy as np

from copy import deepcopy


def parse_args():
	"""This function creates argument parser and parses the scrip input arguments.
	This is the most common way to define input arguments in python. It is used
	by train.py and human_eval.py

	To change the parameters, pass them to the script, for example:

	python cli/train.py \
		--output_dir output_dir \
		--weight_decay 0.01
	
	Default arguments have the meaning of being a reasonable default value, not of the last arguments used.
	"""
	parser = argparse.ArgumentParser(description="Train machine translation transformer model")

	# Required arguments
	parser.add_argument(
		"--output_dir",
		type=str,
		default="Outputs",
		help=("Where to store the final model. "
			  "Should contain the tokenizer in the following format: "
			  r"output_dir/tokenizer_name. "
			  "It should be a directory containing a tokenizer.json file."
		),
	)
	parser.add_argument(
		"--tokenizer_name",
		type=str,
		default="Salesforce/codet5-base",
		help="Name of tokenizer to be used.",
	)
	parser.add_argument(
		"--pretrained_model_name",
		type=str,
		default="Salesforce/codet5-base",
		help="Name of pretrained model to be used.",
	)
	parser.add_argument(
		"--dataset_name",
		type=str,
		default="wikisql",
		help="Name of dataset to be used for fine-tuning.",
	)
	parser.add_argument(
		"--train_from_scratch",
		default=False,
		action="store_true",
		help="Whether to train the model from scratch or not.",
	)
	parser.add_argument(
		"--debug",
		default=False,
		action="store_true",
		help="Whether to use a small subset of the dataset for debugging.",
	)
	# Model arguments
	parser.add_argument(
		"--num_layers",
		default=6,
		type=int,
		help="Number of hidden layers in the Transformer encoder",
	)
	parser.add_argument(
		"--hidden_size",
		default=512,
		type=int,
		help="Hidden size of the Transformer encoder",
	)
	parser.add_argument(
		"--num_heads",
		default=8,
		type=int,
		help="Number of attention heads in the Transformer encoder",
	)
	parser.add_argument(
		"--fcn_hidden",
		default=2048,
		type=int,
		help="Hidden size of the FCN",
	)
	parser.add_argument(
		"--max_seq_length",
		type=int,
		default=128,
		help="The maximum total sequence length for source and target texts after "
		"tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
		"during ``evaluate`` and ``predict``.",
	)
	parser.add_argument(
		"--src_max_len",
		type=int,
		default=128,
		help="The maximum total sequence length for source texts after "
		"tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
		"during ``evaluate`` and ``predict``.",
	)
	parser.add_argument(
		"--tgt_max_len",
		type=int,
		default=128,
		help="The maximum total sequence length for target texts after "
		"tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
		"during ``evaluate`` and ``predict``.",
	)
	parser.add_argument(
		"--preprocessing_num_workers",
		type=int,
		default=8,
		help="The number of processes to use for the preprocessing.",
	)
	parser.add_argument(
		"--overwrite_cache",
		type=bool,
		default=None,
		help="Overwrite the cached training and evaluation sets",
	)

	# Training arguments
	parser.add_argument(
		"--device",
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Device (cuda or cpu) on which the code should run",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=8,
		help="Batch size (per device) for the training dataloader.",
	)
	parser.add_argument(
		"--learning_rate",
		type=float,
		default=5e-5,
		help="Initial learning rate (after the potential warmup period) to use.",
	)
	parser.add_argument(
		"--weight_decay",
		type=float,
		default=0.0,
		help="Weight decay to use.",
	)
	parser.add_argument(
		"--dropout_rate",
		default=0.1,
		type=float,
		help="Dropout rate of the Transformer encoder",
	)
	parser.add_argument(
		"--num_train_epochs",
		type=int,
		default=1,
		help="Total number of training epochs to perform.",
	)
	parser.add_argument(
		"--eval_every_steps",
		type=int,
		default=5000,
		help="Perform evaluation every n network updates.",
	)
	parser.add_argument(
		"--logging_steps",
		type=int,
		default=10,
		help="Compute and log training batch metrics every n steps.",
	)
	parser.add_argument(
		"--max_train_steps",
		type=int,
		default=None,
		help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
	)
	parser.add_argument(
		"--lr_scheduler_type",
		type=transformers.SchedulerType,
		default="linear",
		help="The scheduler type to use.",
		choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
	)
	parser.add_argument(
		"--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
	)
	parser.add_argument(
		"--generation_type",
		choices=["greedy", "beam_search"],
		default="beam_search",
	)
	parser.add_argument(
		"--beam_size",
		type=int,
		default=5,
		help=("Beam size for beam search generation. "
			  "Decreasing this parameter will make evaluation much faster, "
			  "increasing this (until a certain value) would likely improve your results."
		),
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="A seed for reproducible training.",
	)
	parser.add_argument(
		"--wandb_project", 
		default="sql_to_text",
		help="wandb project name to log metrics to"
	)

	args = parser.parse_args()
	
	return args


def sample_small_debug_dataset(raw_datasets, sample_size):
	"""Samples portion of a HuggingFace dataset to be used for debugging.

	Args:
		raw_datasets: HuggingFace dataset to be sampled
		sample_size: int, sample size to take from dataset
	
	Returns:
		HuggingFace dataset that has been sampled
	"""
	random_indices = random.sample(list(range(len(raw_datasets["train"]))), sample_size)
	subset = raw_datasets["train"].select(random_indices)
	raw_datasets["train"] = deepcopy(subset)
	if "validation" in raw_datasets:
		raw_datasets["validation"] = deepcopy(subset)
	if "test" in raw_datasets:
		raw_datasets["test"] = deepcopy(subset)
	return raw_datasets


def sample_human_eval_dataset(raw_datasets, sample_size, set_to_sample):
	"""Samples portion of a HuggingFace dataset to be used for human evaluation.

	Args:
		raw_datasets: HuggingFace dataset to be sampled
		sample_size: int, sample size to take from dataset
		set_to_sample: str, set to sample from (i.e. train/val/test)
	
	Returns:
		HuggingFace dataset that has been sampled
	"""
	if set_to_sample not in raw_datasets:
		set_to_sample = 'train'

	random_indices = random.sample(list(range(len(raw_datasets[set_to_sample]))), sample_size)
	eval_samples = raw_datasets[set_to_sample].select(random_indices)
	return eval_samples


def pad(sequence_list, pad_id):
	"""Pads sequence_list to the longest sequence in the batch with pad_id.

	Args:
		sequence_list: a list of size batch_size of numpy arrays of different length
		pad_id: int, a pad token id
	
	Returns:
		torch.LongTensor of shape [batch_size, max_sequence_len]
	"""
	max_len = max(len(x) for x in sequence_list)
	padded_sequence_list = []
	for sequence in sequence_list:
		padding = [pad_id] * (max_len - len(sequence))
		padded_sequence = sequence + padding
		padded_sequence_list.append(padded_sequence)

	return torch.LongTensor(padded_sequence_list)


def combine_subsets(dataset, header_key, inner_key=None):
	"""Combines subsets (train/test/val) of a HuggingFace dataset for a specfic data field

	Args:
		dataset: HuggingFace dataset to combine
		header_key: str, key value to use for dataset field (i.e. dataset[header_key])
		inner_key: str, optional, key value to use as inner key (i.e. dataset[header_key][inner_key])

	Returns:
		list of combined dataset fields
	"""
	splits = set(dataset.keys()).intersection(['train', 'validation', 'test'])
	all_data = []

	for subset in splits:
		data = dataset[subset][header_key]
		if inner_key:
			data = [ex[inner_key] for ex in data]
		all_data += data

	return all_data


def get_max_seq_len(dataset, header_key, inner_key=None, max_type='longest', pct=50):
	"""Gets the largest sequence length across an entire HuggingFace dataset (train/val/test)
	  	using different max value sampling methods

	Args:
		dataset: HuggingFace dataset to use
		header_key: str, key value to use for dataset field (i.e. dataset[header_key])
		inner_key: str, optional, key value to use as inner key (i.e. dataset[header_key][inner_key])
		max_type: str, optional, how to sample the dataset for max value
					- 'longest' (default): Returns longest value in dataset field
					- 'average': Returns the average length in dataset field
					- 'percentile': Returns length that percentage of data items have
		pct: int, optional, percentile value to use if max_type='percentile'
					- Ex, if pct=99, the seq len corresponding to the 99th percentile is returned

	Returns:
		max sequence length for a dataset field
	"""
	all_data = combine_subsets(dataset=dataset, header_key=header_key, inner_key=inner_key)
	all_lens = list(map(len, all_data))

	max_funcs = {
		'longest': lambda d: max(d),
		'average': lambda d: sum(d) // len(d) + 1,
		'percentile': lambda d: np.percentile(d, pct, interpolation='higher')
	}
	if max_type not in max_funcs.keys():
		max_type = 'longest'

	return max_funcs[max_type](all_lens)
