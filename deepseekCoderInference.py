import modal
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import pathlib

MODEL_DIR = "/model"
MODEL_NAME = "agentica-org/DeepCoder-14B-Preview"
N_GPU = 1

roasting_prompt = """You are a brutally honest, sarcastic, and witty senior software engineer. Your job is to roast people's LeetCode code solutions.

You're given a code snippet. Your job is to roast it in a funny, clever, and mildly insulting way — like a snarky code reviewer on a bad day.

Be playful, creative, and don't hold back. Make comments about bad logic, unnecessary complexity, cringe variable names, poor edge case handling, or anything else that deserves it.

Here's the code to roast:"""

test_code = """
    class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap = {}

        for i, num in enumerate(nums):
            complement = target - num
            if complement in hashmap:
                return [hashmap[complement],i]
            hashmap[num] = i
        return -1
    """
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.7.2",
        "huggingface_hub[hf_transfer]==0.26.2",

    )
    .env({"HF_HUB_CACHE": "/root/.cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
    # .env({"VLLM_USE_V1": "1"})  # use v1 API for VLLM
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App(
    "CodeRoaster",
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,},
)

class RoastRequest(BaseModel):
    code: str

@app.cls(
    gpu=f"H100:{N_GPU}",  # Try using an A100 or H100 if you've got a large model or need big batches!
    max_containers=10,  # default max GPUs for Modal's free tier
)
class Model:
    @modal.enter()
    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        print("📣 Model loaded")
    
    @modal.batched(max_batch_size=1, wait_ms=100)
    def generate(self, code):
        roast_prompt = roasting_prompt + "\n\n" + code[0]
        messages = [
            {"role": "user", "content": roast_prompt},
        ]
        print("📣 Generating text...")

        generation_args = {
            "max_new_tokens": 64000,
            "temperature": 0.6,
            "top_p": 0.95,
        }

        output = self.pipe(messages, **generation_args)
        print("Output:", output)
        return output
    
@app.function(gpu="h100")
async def generate_text(prompt: str) -> str:
    model = Model()
    return model.generate.remote(prompt)

@app.function(gpu="h100")
@modal.fastapi_endpoint(method="POST", docs=True)
def generate_text_endpoint(request: RoastRequest):
    """
    Generate text using the Phi-3.5 model.
    """
    code = request.code
    print(code)
    print(f"Received code to roast :\n{code}")
    response = generate_text.remote(code)
    print(f"Generated response: {response}")
    return response