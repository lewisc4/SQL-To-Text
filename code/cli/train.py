# Python standard library imports
import logging
import math
import os
import random
from functools import partial
from packaging import version

# Import from third party libraries
import wandb
import transformers
import datasets
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, RobertaTokenizer, T5ForConditionalGeneration, Adafactor
from datasets import load_dataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

# Imports from sql_to_text module
from sql_to_text.utils import parse_args, sample_small_debug_dataset, pad, get_max_seq_len


# Setup logging
logger = logging.getLogger(__file__)
logging.basicConfig(
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	datefmt="%m/%d/%Y %H:%M:%S",
	level=logging.INFO,
)

datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_warning()

# To compute BLEU we will use Huggingface Datasets implementation of it
# Sacrebleu is a flavor of BLEU that standardizes some of the BLEU parameters.
bleu = datasets.load_metric("sacrebleu")


def preprocess_function(examples, src_max_len, tgt_max_len, tokenizer):
	"""Tokenize, truncate and add special tokens to the examples. Shift the target text by one token.
	
	Args:
		examples: A dictionary with a single key "translation",
			which is a list of dictionaries with keys corresponding to queryies and questions.

		src_max_len: The maximum total sequence length (in tokens) for source texts.
		tgt_max_len: The maximum total sequence length (in tokens) for target texts.
		source_tokenizer: The tokenizer to use for the source and target texts.
	"""
	inputs = [ex['human_readable'] for ex in examples['sql']]
	targets = examples['question']
	# Apply tokenizer to raw model inputs
	model_inputs = tokenizer(
		inputs, padding='longest', max_length=src_max_len, 
		truncation=True, return_tensors='pt'
	)
	# Apply tokenizer to raw model targets
	target_encoding = tokenizer(
		targets, padding='longest', max_length=tgt_max_len, 
		truncation=True
	)
	labels = target_encoding.input_ids
	
	model_inputs['input_ids'] = model_inputs['input_ids'].tolist()
	model_inputs['attention_mask'] = model_inputs['attention_mask'].tolist()
	model_inputs['labels'] = labels

	return model_inputs


def collation_function_for_seq2seq(batch, pad_token_id):
	"""
	Args:
		batch: a list of dicts of numpy arrays with keys
			input_ids
			attention_mask
			labels
	"""
	input_ids_list = [ex['input_ids'] for ex in batch]
	attention_mask_list = [ex['attention_mask'] for ex in batch]
	labels_list = [ex['labels'] for ex in batch]
	
	collated_batch = {
		'input_ids': pad(input_ids_list, pad_token_id),
		'attention_mask': pad(attention_mask_list, pad_token_id),
		'labels': pad(labels_list, pad_token_id),
	}

	return collated_batch


def compute_accuracy(logits, labels, pad_id):
	"""Computes word accuracy, not very useful for SQL-to-text, but can help identify bugs
	
	Args:
		logits: Logits from the model's output on training batch
		labels: Training batch labels fed to model
		pad_id: Padding token id used by the tokenizer
	"""
	predictions = logits.argmax(-1)
	label_nonpad_mask = labels != pad_id
	num_words_in_batch = label_nonpad_mask.sum().item()

	accuracy = (predictions == labels).masked_select(label_nonpad_mask).sum().item() / num_words_in_batch
	return accuracy


def evaluate_model(model, dataloader, tokenizer, device, max_seq_length, num_beams):
	"""Evaluate a model using the model's generate function, which applies beam searching
	
	Args:
		model: the model to evaluate
		dataloader: The dataloader to use for evaluation data
		tokenizer: The tokenizer used to tokenize the source and target texts
		device: The device to evaluate on
		max_seq_length: The maximum total sequence length (in tokens) for target texts.
		num_beams: The beam size to use when generate() is called i.e. inference
	"""
	n_generated_tokens = 0
	model.eval()

	# Iterate through each batch of eval data
	for batch in tqdm(dataloader, desc='Evaluation'):
		# Use inference mode so we do not update our gradients
		with torch.inference_mode():
			# Transfer batch data to training device
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)

			# Get model predictions/output tokens
			generated_tokens = model.generate(
				input_ids,
				attention_mask=attention_mask,
				max_length=max_seq_length,
				num_beams=num_beams
			).to(device)
			decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
			decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

			# Get the number of tokens in the (decoded) model predictions
			for pred in decoded_preds:
				n_generated_tokens += len(tokenizer(pred)['input_ids'])

			decoded_preds = [pred.strip() for pred in decoded_preds]
			decoded_labels = [[label.strip()] for label in decoded_labels]
			# Add the current batch of predictions/true labels to compute bleu score
			bleu.add_batch(predictions=decoded_preds, references=decoded_labels)

	# Convert model back to training mode
	model.train()
	# Calculate bleu score and log eval results
	eval_metric = bleu.compute()
	evaluation_results = {
		'bleu': eval_metric['score'],
		'generation_length': n_generated_tokens / len(dataloader.dataset)
	}
	return evaluation_results, input_ids, decoded_preds, decoded_labels


def train_model(model, tokenizer, train_dataloader, eval_dataloader, optimizer, lr_scheduler, args):
	"""Train (fine-tune) a model
	
	Args:
		model: the model to train
		tokenizer: The tokenizer used to tokenize the source and target texts
		train_dataloader: The dataloader to use for training data
		eval_dataloader: The dataloader to use for evaluation data
		optimizer: The optimizer to use for training
		lr_scheduler: The learning rate scheduler to use for training
		args: The set of arguments used to run this training script
	"""
	progress_bar = tqdm(range(args.max_train_steps))
	global_step = 0

	# Train for the desired number of training epochs
	for epoch in range(args.num_train_epochs):
		# Set the model to training mode (ensures gradients are updated)
		model.train()
		# Iterate through each batch of training data
		for batch in train_dataloader:
			# Transfer batch data to training device
			input_ids = batch['input_ids'].to(args.device)
			attention_mask = batch['attention_mask'].to(args.device)
			labels = batch['labels'].to(args.device)
			# Set the padding token to -100 in the training labels
			labels[labels == tokenizer.pad_token_id] = -100

			# Generate model predictions
			model_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

			logits = model_output.logits.to(args.device)
			loss = model_output.loss.to(args.device)
			
			# Progress/update our model and schedulers/optimizers
			loss.backward()
			optimizer.step()
			lr_scheduler.step()
			optimizer.zero_grad()

			progress_bar.update(1)
			global_step += 1

			# Log training loss, lr, and current epoch for each batch
			wandb.log(
				{
					'train_loss': loss,
					'learning_rate': optimizer.param_groups[0]['lr'],
					'epoch': epoch,
				},
				step=global_step,
			)

			# Logs training accuracy (i.e. word accuracy) every n steps
			if global_step % args.logging_steps == 0:
				accuracy = compute_accuracy(logits=logits, labels=labels, pad_id=tokenizer.pad_token_id)

				wandb.log(
					{'train_batch_word_accuracy': accuracy},
					step=global_step,
				)

			# Evaluates model every n steps or at the very end of training (last step)
			if global_step % args.eval_every_steps == 0 or global_step == args.max_train_steps:
				eval_results, last_input_ids, last_decoded_preds, last_decoded_labels = evaluate_model(
					model=model,
					dataloader=eval_dataloader,
					tokenizer=tokenizer,
					device=args.device,
					max_seq_length=args.tgt_max_len,
					num_beams=args.beam_size,
				)

				wandb.log(
					{
						'eval/bleu': eval_results['bleu'],
						'eval/generation_length': eval_results['generation_length'],
					},
					step=global_step,
				)
				logger.info('Generation example:')
				random_index = random.randint(0, len(last_input_ids) - 1)
				logger.info(f'Input sentence: {tokenizer.decode(last_input_ids[random_index], skip_special_tokens=True)}')
				logger.info(f'Generated sentence: {last_decoded_preds[random_index]}')
				logger.info(f'Reference sentence: {last_decoded_labels[random_index][0]}')

				logger.info('Saving model checkpoint to %s', args.output_dir)
				model.save_pretrained(args.output_dir)

			if global_step >= args.max_train_steps:
				break

	# Return the fine-tuned model
	return model


def main():
	# Parse training arguments
	args = parse_args()

	logger.info(f'Starting script with arguments: {args}')

	# Initialize wandb as soon as possible to log all stdout to the cloud 
	wandb.init(project=args.wandb_project, config=args)
	# Make sure output directory exists, if not create it
	os.makedirs(args.output_dir, exist_ok=True)

	# pad_token_id=0, bos_token_id=1, eos_token_id=2
	tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)

	# If we are training a model from scratch or not
	if args.train_from_scratch:
		config = AutoConfig.from_pretrained(args.pretrained_model_name)
		# Below does not load the weights, just parameters because we are not using from_pretrained()
		# See: https://huggingface.co/transformers/v3.0.2/model_doc/t5.html#transformers.T5ForConditionalGeneration
		model = T5ForConditionalGeneration(config)
	else:
		# Load the pre-trained model using the provided model checkpoint name
		model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_name)
	# Move model to device for training
	model.to(device=args.device)

	# Load the dataset
	dataset = load_dataset(args.dataset_name)

	# Sample dataset for debugging, if we are using debug mode
	if args.debug:
		dataset = sample_small_debug_dataset(dataset, sample_size=100)

	# (SQL) Longest: 468, Mean: 64, Median: 60, 98th Percentile: 112
	# Find max seq length for source (SQL) texts, use length corresponding to 99th percentile
	args.src_max_len = get_max_seq_len(
		dataset=dataset, header_key='sql', inner_key='human_readable', 
		max_type='percentile', pct=99
	)

	# (Text) Longest: 275, Mean: 62, Median: 57, 98th Percentile: 123
	# Find max seq length for target (questions) texts, use length corresponding to 99th percentile
	args.tgt_max_len = get_max_seq_len(
		dataset=dataset, header_key='question',
		max_type='percentile', pct=99
	)

	# Don't use large sequence lengths when debugging, too slow for no reason
	if args.debug:
		args.src_max_len = 8
		args.tgt_max_len = 8

	# Set the max sequence length between the source AND target texts
	args.max_seq_length = max(args.src_max_len, args.tgt_max_len)

	# First we tokenize all the texts.
	column_names = dataset['train'].column_names

	# Because .map expects the pre-processing function only to have one argument,
	# we need to wrap preprocess_function() in a partial and provide the rest of the arguments.
	# It is better to do this instead of defining a function right here (as we did in the previous homework)
	preprocess_function_wrapped = partial(
		preprocess_function,
		src_max_len=args.src_max_len,
		tgt_max_len=args.tgt_max_len,
		tokenizer=tokenizer
	)
	processed_datasets = dataset.map(
		preprocess_function_wrapped,
		batched=True,
		num_proc=args.preprocessing_num_workers,
		remove_columns=column_names,
		load_from_cache_file=not args.overwrite_cache,
		desc='Running tokenizer on dataset'
	)

	train_dataset = processed_datasets['train']
	eval_dataset = processed_datasets['validation'] if 'validation' in processed_datasets else processed_datasets['test']
	
	# Log a few random samples from the training set:
	for index in random.sample(range(len(train_dataset)), 2):
		logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
		logger.info(f"Decoded input_ids: {tokenizer.decode(train_dataset[index]['input_ids'])}")
		logger.info(f"Decoded labels: {tokenizer.decode(train_dataset[index]['labels'])}")
		logger.info("\n")

	collation_function_for_seq2seq_wrapped = partial(
		collation_function_for_seq2seq,
		pad_token_id=tokenizer.pad_token_id
	)

	train_dataloader = DataLoader(
		train_dataset, shuffle=True,
		collate_fn=collation_function_for_seq2seq_wrapped,
		batch_size=args.batch_size
	)

	eval_dataloader = DataLoader(
		eval_dataset,
		collate_fn=collation_function_for_seq2seq_wrapped,
		batch_size=args.batch_size
	)

	optimizer = Adafactor(
		model.parameters(),
		scale_parameter=False,
		relative_step=False,
		warmup_init=False,
		lr=args.learning_rate,
		weight_decay=args.weight_decay,
	)

	# Scheduler and math around the number of training steps.
	num_update_steps_per_epoch = len(train_dataloader)
	if args.max_train_steps is None:
		args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
	else:
		args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

	lr_scheduler = transformers.get_scheduler(
		name=args.lr_scheduler_type,
		optimizer=optimizer,
		num_warmup_steps=args.num_warmup_steps,
		num_training_steps=args.max_train_steps,
	)

	logger.info("***** Running training *****")
	logger.info(f"  Num examples = {len(train_dataset)}")
	logger.info(f"  Num Epochs = {args.num_train_epochs}")
	logger.info(f"  Total optimization steps = {args.max_train_steps}")

	# Log a pre-processed training example to make sure the pre-processing does not have bugs in it
	# and we do not input garbage to our model.
	batch = next(iter(train_dataloader))
	logger.info("Look at the data that we input into the model, check that it looks like what we expect.")
	for index in random.sample(range(len(batch)), 2):
		logger.info(f"Decoded input_ids: {tokenizer.decode(batch['input_ids'][index])}")
		logger.info(f"Decoded labels: {tokenizer.decode(batch['labels'][index])}")
		logger.info("\n")
	
	# Fine-tune the pre-trained model
	finetuned_model = train_model(
		model=model, tokenizer=tokenizer,
		train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
		optimizer=optimizer, lr_scheduler=lr_scheduler,
		args=args
	)

	logger.info("Saving final model checkpoint to %s", args.output_dir)
	finetuned_model.save_pretrained(args.output_dir)
	
	logger.info("Uploading tokenizer, model and config to wandb")
	wandb.save(os.path.join(args.output_dir, "*"))
	
	logger.info(f"Script finished succesfully, model saved in {args.output_dir}")


if __name__ == '__main__':
	main()
