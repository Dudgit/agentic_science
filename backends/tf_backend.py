from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TFBackend:
    def __init__(self, model_name="Qwen/Qwen3.6-35B-A3B"):
        print(f"Loading {model_name} onto A2 Cluster GPUs...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # This is the critical line! device_map="auto" forces it onto the GPUs
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            max_memory={0: "14GiB",1: "14GiB",2: "14GiB",3: "14GiB"},
            #load_in_4bit=True, 
            torch_dtype=torch.float16, # Keeps it in standard half-precision
            attn_implementation="eager",
            trust_remote_code=True
        )
        print("Model successfully loaded into VRAM.")
        
    def generate(self, messages):
        # 1. Format the conversation
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 2. Move inputs to the SAME GPU as the first model layer
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = self.model.get_input_embeddings().weight.device

        inputs = {k: v.to(device)for k, v in inputs.items()}
                
        # 3. Generate
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=512,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None
        )
        
        # 4. Decode just the new generated text
        input_length = inputs['input_ids'].shape[1]
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return response