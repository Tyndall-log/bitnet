from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	TrainingArguments,
	Trainer,
	DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model, TaskType


def build_text(tokenizer, instruction: str, output: str) -> str:
	# 너가 쓰던 chat template 흐름을 그대로 사용
	messages = [
		{"role": "system", "content": "You are a helpful AI assistant."},
		{"role": "user", "content": instruction.strip()},
		{"role": "assistant", "content": output.strip()},
	]
	return tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=False,  # 학습 데이터는 답까지 포함
	)


def find_lora_targets(model):
	# BitNet이 LLaMA 계열 네이밍을 따를 가능성이 높아서 우선 이걸로 시도
	candidates = set()
	for name, _ in model.named_modules():
		if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
			candidates.add(name.split(".")[-1])
	# candidates는 모듈 "클래스"가 아니라 "속성명" 기준으로 넣어야 해서 split[-1] 사용
	# 일반적으로 아래처럼 들어감: {"q_proj","k_proj","v_proj","o_proj",...}
	if len(candidates) == 0:
		# 혹시 네이밍이 다르면, proj 들어간 애들로 fallback
		for name, _ in model.named_modules():
			last = name.split(".")[-1]
			if "proj" in last:
				candidates.add(last)
	return sorted(list(candidates))


def main():
	model_id = "microsoft/bitnet-b1.58-2B-4T-bf16"

	root_dir = Path(".")
	cache_dir = root_dir / "data" / "hf_cache"
	cache_dir.mkdir(parents=True, exist_ok=True)

	device = "cuda" if torch.cuda.is_available() else (
		"mps" if torch.backends.mps.is_available() else "cpu"
	)

	tokenizer = AutoTokenizer.from_pretrained(
		model_id,
		cache_dir=cache_dir,
	)

	# (중요) pad_token이 없으면 Trainer에서 padding 시 문제날 수 있어
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		dtype=torch.bfloat16,
		cache_dir=cache_dir,
	)

	# LoRA 설정
	target_modules = find_lora_targets(model)
	print("LoRA target_modules =", target_modules)

	lora_config = LoraConfig(
		task_type=TaskType.CAUSAL_LM,
		r=8,
		lora_alpha=16,
		lora_dropout=0.05,
		bias="none",
		target_modules=target_modules,
	)

	model = get_peft_model(model, lora_config)
	model.print_trainable_parameters()

	model.to(device)

	# 데이터셋 로드
	ds = load_dataset(
		"BCCard/BCCard-Finance-Kor-QnA",
		cache_dir=(cache_dir / "datasets").as_posix(),
	)

	# 토크나이즈
	max_length = 1024

	def preprocess(batch):
		texts = []
		for ins, out in zip(batch["instruction"], batch["output"]):
			texts.append(build_text(tokenizer, ins, out))

		enc = tokenizer(
			texts,
			max_length=max_length,
			truncation=True,
			padding=False,  # collator가 해줌
			return_tensors=None,
		)

		# Causal LM 학습: labels = input_ids (패딩은 collator에서 -100 처리)
		enc["labels"] = enc["input_ids"].copy()
		return enc

	train_ds = ds["train"].map(
		preprocess,
		batched=True,
		remove_columns=ds["train"].column_names,
	)

	# Collator: LM용 패딩 + labels에서 pad를 -100으로 마스킹
	data_collator = DataCollatorForLanguageModeling(
		tokenizer=tokenizer,
		mlm=False,
	)

	# 학습 설정
	out_dir = root_dir / "outputs" / "bitnet_lora_bccard"
	out_dir.mkdir(parents=True, exist_ok=True)

	training_args = TrainingArguments(
		output_dir=out_dir.as_posix(),
		per_device_train_batch_size=1,
		gradient_accumulation_steps=16,
		learning_rate=2e-4,
		num_train_epochs=1,
		warmup_ratio=0.03,
		logging_steps=10,
		save_steps=200,
		save_total_limit=2,
		bf16=torch.cuda.is_available(),  # cuda일 때만 bf16 trainer 경로가 안정적
		fp16=False,
		gradient_checkpointing=True,
		report_to="none",
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_ds,
		data_collator=data_collator,
	)

	trainer.train()

	# LoRA 어댑터 저장
	model.save_pretrained((out_dir / "lora_adapter").as_posix())
	tokenizer.save_pretrained((out_dir / "lora_adapter").as_posix())

	print("Done! Saved to:", (out_dir / "lora_adapter").as_posix())


if __name__ == "__main__":
	main()