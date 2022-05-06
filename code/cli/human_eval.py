# Python standard library imports
import random
from os.path import isfile, join

# Import from third party libraries
import datasets
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from scipy.stats.stats import pearsonr   

# Imports from sql_to_text module
from sql_to_text.utils import parse_args, sample_human_eval_dataset


# To compute BLEU we will use Huggingface Datasets implementation of it
# Sacrebleu is a flavor of BLEU that standardizes some of the BLEU parameters.
bleu = datasets.load_metric('sacrebleu')


def get_model_generations(args, save_name, sample_size):
	"""Generate model predictions sampled from a dataset and save them for human eval
	
	Args:
		args: The command line arguments used to run this script
		save_name: The file name to save the generation results to
		sample_size: Number of samples to generate predictions for
	"""
	# Load the dataset (default of 'wikisql')
	dataset = load_dataset(args.dataset_name)
	# Load the tokenizer (default of 'Salesforce/codet5-base')
	tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
	# Load the fine-tuned model, stored in args.output_dir (same it was trained and saved to)
	model = T5ForConditionalGeneration.from_pretrained(args.output_dir)

	# Sample examples test examples for human eval
	human_eval_data = sample_human_eval_dataset(raw_datasets=dataset, sample_size=sample_size, set_to_sample='test')

	# Get the input (SQL) and target (questions) texts
	inputs = [ex['human_readable'] for ex in human_eval_data['sql']]
	targets = human_eval_data['question']
	generated_samples = []
	# Generate sequences for each (input, target) pair
	for i, (query, question) in enumerate(zip(inputs, targets)):
		print(str(i + 1) + '/' + str(sample_size))
		input_ids = tokenizer(query, max_length=len(query), return_tensors='pt').input_ids
		output_ids = model.generate(input_ids, max_length=len(question), num_beams=8)
		decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
		generated_samples.append(decoded_output)

	# Data to save in dict format
	data_to_eval = {
		'SQL Query': inputs,
		'Actual Question': targets,
		'Predicted Question': generated_samples,
	}

	# Store data in Pandas DF and save to file
	df = pd.DataFrame.from_dict(data_to_eval, orient='index').transpose()
	df.to_csv(save_name, index=False)


def get_bleu_correlations(ratings_data):
	"""Get BLEU score for generated sequences and find correlations between BLEU human rating
	
	Args:
		ratings_data: Pandas DF, the data to use (loaded from csv in args.output_dir)
	"""
	# Get the actual, predicted, and ratings data as lists
	references = ratings_data['Actual Question'].tolist()
	predictions = ratings_data['Predicted Question'].tolist()
	ratings = ratings_data['Rating'].tolist()
	ratings = list(map(int, ratings))

	# Compute BLEU score for each (reference, prediction) pair
	scores = []
	for ref, pred in zip(references, predictions):
		p = [[pred.strip()]]
		r = [[ref.strip()]]

		bleu_score = bleu.compute(predictions=p, references=r)
		scores.append(bleu_score['score'])

	# Add BLEU score to input DF for plotting
	ratings_data['BLEU Score'] = scores
	# Get Pearson correlation
	r, p = pearsonr(scores, ratings)

	# Plot and show results
	g = sns.JointGrid(x='Rating', y='BLEU Score', data=ratings_data)
	g.plot_joint(sns.regplot)
	g.ax_joint.annotate(f'$pearsonr = {r:.5f}, p = {p:.5f}$',
						xy=(0.1, 0.9), xycoords='axes fraction',
						ha='left', va='center',
						bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
	g.fig.tight_layout()
	g.fig.suptitle('Human Rating vs. BLEU Score')
	plt.show()


if __name__ == '__main__':
	# Set the random seed to reproduce results
	random.seed(1234)
	# Command line args used to run script
	args = parse_args()

	# File name to be used for saving/loading eval data
	eval_fname = join(args.output_dir,'human_eval_data.csv')

	# If the file exists and has human ratings, get the BLEU correlations
	# Otherwise, generate model predictions on 110 data samples 
	if isfile(eval_fname):
		ratings_data = pd.read_csv(eval_fname)
		# No ratings made yet
		if 'Rating' not in ratings_data.columns:
			raise RuntimeError('No human-generated ratings have been made yet')		
		get_bleu_correlations(ratings_data=ratings_data)
	else:
		get_model_generations(args, save_name=eval_fname, sample_size=110)
