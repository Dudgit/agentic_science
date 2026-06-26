from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os


class HFBackend:
    def __init__(
        self,
        model_name,
        gpu_ids=None,
        dtype=torch.float16,
        max_new_tokens=512
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        # -----------------------------
        # 1. GPU SCOPE CONTROL
        # -----------------------------
        if gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        # IMPORTANT: reset visible devices AFTER setting env
        torch.cuda.empty_cache()

        # -----------------------------
        # 2. TOKENIZER
        # -----------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # -----------------------------
        # 3. GPU MEMORY PLAN (IMPORTANT)
        # -----------------------------
        # Map all layers onto visible GPUs automatically BUT CONTROLLED
        max_memory = None
        if gpu_ids is not None:
            max_memory = {
                    0: "12GB",
                    1: "12GB"
                }

        # -----------------------------
        # 4. MODEL LOAD
        # -----------------------------
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",          # SAFE now because GPUs are scoped
            torch_dtype=dtype,
            trust_remote_code=True,
            max_memory=max_memory
        )

        self.model.eval()

    # -----------------------------
    # GENERATION
    # -----------------------------
    def generate(self, messages, max_new_tokens=None):

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")

        device = self.model.get_input_embeddings().weight.device

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(out[0], skip_special_tokens=True)