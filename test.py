from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/bitnet-b1.58-2B-4T-bf16"
cache_dir = Path("./data/hf_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
	model_id,
	cache_dir=cache_dir,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
	cache_dir=cache_dir,
)

model.eval()
device = "cuda" if torch.cuda.is_available() else (
	"mps" if torch.backends.mps.is_available() else "cpu"
)
model.to(device)

# Apply the chat template
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "한국 법률에 따르면, 계약서 작성 시 어떤 사항들을 반드시 포함해야 하나요?"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
chat_outputs = model.generate(**chat_input, max_new_tokens=50)
response = tokenizer.decode(chat_outputs[0][chat_input['input_ids'].shape[-1]:], skip_special_tokens=True) # Decode only the response part
print("\nAssistant Response:", response)
