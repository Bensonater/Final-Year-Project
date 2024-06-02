from model.qgpt2_models import MultiHeadsQGPT2Model, SingleHeadQGPT2Model
from transformers import GPT2Tokenizer, GPT2ForQuestionAnswering
import time

from logging import getLogger, ERROR
getLogger("transformers.modeling_utils").setLevel(ERROR)


model = SingleHeadQGPT2Model.from_pretrained("gpt2", n_bits=5, use_cache=False)
model.set_fhe_mode(fhe="disable", true_float=False)

# model = GPT2ForQuestionAnswering.from_pretrained("openai-community/gpt2")

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

# user_input = input("Your thoughts here: ")

user_input = """

You are performing a sentiment analysis task. Read the below reviews and deicde the sentiment. Only respond in either Positive, Neutral, or Negative.

Review: "The packaging was really nice and the product works perfectly. Will buy again."
Sentiment: Positive

Review: "The product is decent for the price. Nothing exceptional, but it does the job."
Sentiment: Neutral

Review: "I had an issue with the product but customer service resolved it quickly. Satisfied with the support."
Sentiment: """

# user_input = "hello world"

# input_ids = tokenizer(user_input, return_tensors="pt")

from torch import tensor, topk

tokens = tokenizer.encode(user_input)

input_ids = tensor(tokens).unsqueeze(0)

model.compile(input_ids)
model.set_fhe_mode(fhe="simulate")
max_tokens = 4


start = time.perf_counter()
output = model(input_ids).logits
end = time.perf_counter()

print(f"Encoding {len(tokens)}, Run time: {end - start:.4f} seconds")
