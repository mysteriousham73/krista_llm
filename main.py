import time
from transformers import GPTJForCausalLM, AutoTokenizer, AutoModelForCausalLM
import torch
import re

#model = GPTJForCausalLM.from_pretrained("togethercomputer/GPT-JT-6B-v1", cache_dir="/media/models/.cache/huggingface/hub", torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda:0")
#tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1", cache_dir="/media/models/.cache/huggingface/hub")

model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-6B-nl", cache_dir="/media/models/.cache/huggingface/hub", torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-6B-nl", cache_dir="/media/models/.cache/huggingface/hub")


print(model.get_memory_footprint())
f = open("conditioning_prompt.json", "r")
sample_responses = f.read()
f.close()

t_start = time.perf_counter()

input = "What's the current range to PICSAT?"

gpt_prompt = sample_responses + " is a list of sample JSON responses.  Form the question '" + input + "' into properly formatted JSON as above.  **. {\n \"question\":"

#gpt_prompt = f"make the question {input} into json.  The tag \"question\" should be used to define the question.  \"question_type\" should be one of the following:  ['satellite_state', 'satellite_pass_prediction']"
#gpt_prompt = f"You are a question categorizer and satellite name parser.  satellite_pass_query is a question asking when the satellite will be overhead.  satellite_state_query is a question asking about the position, range, and other state variables pertaining the satellite.  categorize the question <{input}> as either satellite_pass_query or satellite_state_query.  Parse the satellite name.  Make the response of the format: \nSatellite Name: <|SATELLITE_NAME|>\nQuery Type: <|QUERY_TYPE|>.  A:  "
gpt_prompt = "#turn the question <when do you think PICSAT will be around again> into JSON containing only satellite_name and query_type.  query_type should be one of: satellite_pass_prediction, satellite_status"
input_ids = tokenizer(gpt_prompt, return_tensors="pt").input_ids.to("cuda:0")

t_tokenize = time.perf_counter()
gen_tokens = model.generate(input_ids, do_sample=True, repetition_penalty=0.1, temperature=.7, max_new_tokens=500,)
gen_text = tokenizer.batch_decode(gen_tokens)[0]

#gen_text = re.split("\*\*. ", gen_text)

#generated = gen_text[1]

open_brackets = []
close_brackets = []

# for m in re.finditer('{', generated):
#          open_brackets.append(m.start())
#
# for m in re.finditer('}', generated):
#          close_brackets.append(m.end())

#enerated_json = generated[open_brackets[0]:close_brackets[len(close_brackets) - 1]]

t_generate = time.perf_counter()
print(f"Tokenization time: {t_tokenize - t_start}\nGeneration time:  {t_generate - t_tokenize} seconds.")
print(gen_text)
