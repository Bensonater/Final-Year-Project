from model.qgpt2_models import SingleHeadQGPT2Model
from transformers import GPT2Tokenizer
import time
from torch import tensor
from logging import getLogger, ERROR
getLogger("transformers.modeling_utils").setLevel(ERROR)

model = SingleHeadQGPT2Model.from_pretrained("gpt2", n_bits=4, use_cache=False)
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")


user_input = """
In a world where technology reigns supreme, humans struggle to find their place. Amidst the rapid advancements and reliance on artificial intelligence, individuals yearn for a deeper connection to nature and their own humanity,
"""

tokens = tokenizer.encode(user_input)

input_ids = tensor(tokens).unsqueeze(0)

model.compile(input_ids)
model.set_fhe_mode(fhe="simulate")

start = time.perf_counter()
output = model(input_ids).logits
end = time.perf_counter()

print(f"Encoding {len(tokens)} tokens, Run time: {end - start:.4f} seconds")
