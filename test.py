from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare input
prompt = "Test query"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate response
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.5)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)