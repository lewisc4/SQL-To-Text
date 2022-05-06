# Import from third party libraries
import streamlit as st
from transformers import RobertaTokenizer, T5ForConditionalGeneration
# streamlit_ace package is used for code syntax highlighting, etc. (looks nice)
from streamlit_ace import st_ace # Ver. 0.0.4

# Imports from sql_to_text module
from sql_to_text.utils import parse_args


# Looks nicer than standard layout
st.set_page_config(layout="wide")

# Get command line args used to run script
args = parse_args()
# Load tokenizer and model only once, upfront for faster demo down the line)
tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
model = T5ForConditionalGeneration.from_pretrained(args.output_dir)


def apply_sql_to_text(sql_query):
	"""Convert a SQL query to text (question), using loaded model and tokenizer
	
	Args:
		sql_query: The SQL query to convert (from demo textbox)
	"""
	# Remove unnecessary whitespace
	sql_query = sql_query.strip()
	# Tokenize query
	input_ids = tokenizer(
		sql_query,
		max_length=len(sql_query),
		padding='longest',
		truncation=True,
		return_tensors='pt'
	).input_ids
	# Generate and decode
	output_ids = model.generate(input_ids, max_length=130, num_beams=8)
	decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
	# Return decoded translation (must be in tuple format)
	return (decoded_output)


def run_demo():
	"""Run the Streamlit demo

		Example queries to translate:
			SELECT age FROM table WHERE student name = Chris
			SELECT COUNT country id FROM table WHERE country residence > 5000
			SELECT MAX food price FROM table WHERE food origin = Italy
	"""
	st.markdown("<h1 style='text-align: center; color: white;'>SQL-To-Text Demo</h1>", unsafe_allow_html=True)
	col1, col2 = st.columns(2)
	with col1:
		with st.form(key='query_form'):
			raw_code = st_ace(placeholder='Enter SQL Query', language='sql', font_size=16, theme='tomorrow_night_bright')
			submit_code = st.form_submit_button('Execute')
	with col2:
		with st.form(key='result_form'):
			placeholder = st.empty()
			with placeholder.container():
				st_ace(placeholder='Translated SQL Query', theme='tomorrow_night_bright', font_size=16, readonly=True)
			clear_res = st.form_submit_button('Clear')
			if submit_code:
				query_results = apply_sql_to_text(raw_code)
				placeholder.empty()
				with placeholder.container():
					st_ace(value=query_results, theme='tomorrow_night_bright', font_size=16, readonly=True)
					clear_res = clear_res


if __name__ == '__main__':
	run_demo()
