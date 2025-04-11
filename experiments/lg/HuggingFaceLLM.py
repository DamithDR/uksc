import asyncio
from typing import List

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class HuggingFaceLLM:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Wrap model with DataParallel for multi-GPU
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompts: List[str], max_length: int) -> List[dict]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(
            self.device)
        with torch.no_grad():
            outputs = self.model.module.generate(**inputs, max_new_tokens=2048, do_sample=True) if isinstance(
                self.model, nn.DataParallel) else self.model.generate(**inputs, max_new_tokens=2048, do_sample=True)
        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return [{"text": output[len(prompt):].strip()} for prompt, output in zip(prompts, decoded_outputs)]
