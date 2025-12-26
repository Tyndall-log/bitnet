import os
from dataclasses import dataclass
from typing import Dict, List

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	DataCollatorForLanguageModeling,
	Trainer,
	TrainingArguments,
	set_seed,
)

IGNORE_INDEX = -100


def _build_prompt(tokenizer, instruction: str) -> str:
	# 채팅 템플릿이 있으면 그걸 그대로 쓰는 게 제일 안전함 (모델이 기대하는 포맷)
	messages = [
		{"role": "system", "content": "You are a helpful AI assistant."},
		{"role": "user", "content": instruction},
	]
	return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _tokenize_sft_example(tokenizer, instruction: str, output: str, max_length: int) -> Dict[str, List[int]]:
	# prompt(=user까지 + assistant 시작 토큰) + answer 를 하나의 LM 시퀀스로 만들고,
	# label은 "정답 토큰" 부분만 학습되도록 prompt 구간을 -100으로 마스킹
	prompt = _build_prompt(tokenizer, instruction)
	full_text = prompt + output

	prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
	full = tokenizer(
		full_text,
		add_special_tokens=False,
		truncation=True,
		max_length=max_length,
	)

	input_ids = full["input_ids"]
	attention_mask = full["attention_mask"]

	labels = input_ids.copy()
	prompt_len = min(len(prompt_ids), len(labels))

	for i in range(prompt_len):
		labels[i] = IGNORE_INDEX

	return {
		"input_ids": input_ids,
		"attention_mask": attention_mask,
		"labels": labels,
	}


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
	print(OmegaConf.to_yaml(cfg))
	set_seed(int(cfg.seed))

	# dtype
	if cfg.model.dtype == "bfloat16":
		torch_dtype = torch.bfloat16
	elif cfg.model.dtype == "float16":
		torch_dtype = torch.float16
	else:
		torch_dtype = torch.float32

	# tokenizer / model
	tokenizer = AutoTokenizer.from_pretrained(
		cfg.model.id,
		cache_dir=cfg.model.cache_dir,
		use_fast=True,
	)
	if tokenizer.pad_token is None:
		# causal LM 학습 편의용 (모델에 따라 eos를 pad로 두는 게 일반적)
		tokenizer.pad_token = tokenizer.eos_token

	model = AutoModelForCausalLM.from_pretrained(
		cfg.model.id,
		cache_dir=cfg.model.cache_dir,
		dtype=torch_dtype,
		# device="mps",
	)
	model.to(device=cfg.model.device)

	# (선택) gradient checkpointing
	if cfg.train.gradient_checkpointing:
		model.gradient_checkpointing_enable()
		model.config.use_cache = False

	# LoRA 적용
	lora_config = LoraConfig(
		r=int(cfg.lora.r),
		lora_alpha=int(cfg.lora.alpha),
		lora_dropout=float(cfg.lora.dropout),
		bias=str(cfg.lora.bias),
		task_type="CAUSAL_LM",
		target_modules=list(cfg.lora.target_modules),
	)
	model = get_peft_model(model, lora_config)
	model.print_trainable_parameters()

	# dataset
	ds = load_dataset(
		cfg.data.name,
		cache_dir=str(cfg.data.cache_dir),
	)[cfg.data.split]

	max_length = int(cfg.data.max_length)

	def map_fn(ex):
		return _tokenize_sft_example(
			tokenizer=tokenizer,
			instruction=ex["instruction"],
			output=ex["output"],
			max_length=max_length,
		)

	ds = ds.map(
		map_fn,
		remove_columns=ds.column_names,
		desc="Tokenizing SFT dataset",
	)

	# collator (labels 이미 만들었으니 MLM=False)
	collator = DataCollatorForLanguageModeling(
		tokenizer=tokenizer,
		mlm=False,
	)

	# TrainingArguments
	args = TrainingArguments(
		output_dir=str(cfg.train.output_dir),
		num_train_epochs=float(cfg.train.num_train_epochs),
		per_device_train_batch_size=int(cfg.train.per_device_train_batch_size),
		gradient_accumulation_steps=int(cfg.train.gradient_accumulation_steps),
		learning_rate=float(cfg.train.learning_rate),
		weight_decay=float(cfg.train.weight_decay),
		warmup_ratio=float(cfg.train.warmup_ratio),
		logging_steps=int(cfg.train.logging_steps),
		save_steps=int(cfg.train.save_steps),
		save_total_limit=int(cfg.train.save_total_limit),
		bf16=bool(cfg.train.bf16),
		tf32=bool(cfg.train.tf32),
		optim=str(cfg.train.optim),
		max_grad_norm=float(cfg.train.max_grad_norm),
		report_to="none",
	)

	trainer = Trainer(
		model=model,
		args=args,
		train_dataset=ds,
		data_collator=collator,
	)

	trainer.train()

	# LoRA 어댑터만 저장 (가볍고, 추후 merge도 쉬움)
	trainer.model.save_pretrained(str(cfg.train.output_dir))
	tokenizer.save_pretrained(str(cfg.train.output_dir))

	print("Done. Saved to:", cfg.train.output_dir)


if __name__ == "__main__":
	main()