from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# to inference from default GPT2 model:
# model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(torch_device)

# to inference from fine-tuned GPT2 model:
model = AutoModelForCausalLM.from_pretrained('./output', pad_token_id=tokenizer.eos_token_id).to(torch_device)

# what sample text to generate from
model_inputs = tokenizer('I need to cancel my order please', return_tensors='pt').to(torch_device)

# generate 100 tokens
greedy_output = model.generate(**model_inputs, max_new_tokens=100)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
