import asyncio
from typing import List

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from experiments.open_llms import get_chat_template


class HuggingFaceLLM:
    def __init__(self, model_name: str, batch_size):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        chat_template = get_chat_template()
        if chat_template:
            self.tokenizer.chat_template = chat_template
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Wrap model with DataParallel for multi-GPU
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            tokenizer=self.tokenizer,
            trust_remote_code=True
        )

    def generate(self, prompts: List[str], max_length: int) -> List[dict]:

        outputs = self.pipe(
            prompts,
            max_new_tokens=2048,
            temperature=0.1,
            pad_token_id=self.pipe.model.config.eos_token_id,
            num_return_sequences=1,
            do_sample=True,
            batch_size=self.batch_size
        )
        responses = []
        for output in tqdm(outputs, total=len(outputs), desc="extracting label outputs"):
            resp = output[0]["generated_text"][-1]['content'].lower().strip()
            responses.append(resp)

        return responses
