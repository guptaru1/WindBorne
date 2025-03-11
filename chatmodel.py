import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

access_token = "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"
llama_model = "meta-llama/Llama-2-7b-chat-hf"

#notebook_login()

class ChatModel():
    def __init__(self, model):
        self.model = model
        
        # Configure 4-bit quantization instead of 8-bit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Changed from 8-bit to 4-bit
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )


        if "mistral" in self.model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                use_fast=True
            )
            self.generator = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                device_map="auto",
                quantization_config=quantization_config
            )
        elif "phi" in self.model.lower():
            # Use Phi-1 instead of Phi-2 (smaller model)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/phi-1",
                use_fast=True
            )
            self.generator = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-1",
                trust_remote_code=True,
                torch_dtype=torch.float32
            ).to('cpu')
        elif "opt" in self.model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                "facebook/opt-125m",
                use_fast=True
            )
            self.generator = AutoModelForCausalLM.from_pretrained(
                "facebook/opt-125m",
                torch_dtype=torch.float32
            ).to('cpu')
        elif "gptj" in self.model.lower():
            # GPT-J 125M is another fast option
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-j-125M",
                use_fast=True
            )
            self.generator = AutoModelForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-125M",
                torch_dtype=torch.float32
            ).to('cpu')
        elif "tiny-llama" in self.model.lower():
            # TinyLlama is very lightweight and Mac-friendly
            self.tokenizer = AutoTokenizer.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                use_fast=True
            )
            self.generator = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                trust_remote_code=True,
                torch_dtype=torch.float32  # Changed to float32
            ).to('cpu')
        elif "falcon" in self.model.lower():
            # Falcon is another good option for Mac
            self.tokenizer = AutoTokenizer.from_pretrained(
                "tiiuae/falcon-7b-instruct",
                use_fast=True
            )
            self.generator = AutoModelForCausalLM.from_pretrained(
                "tiiuae/falcon-7b-instruct",
                trust_remote_code=True,
                torch_dtype=torch.float32  # Changed to float32
            ).to('cpu')

    def chat(self, system_prompt, user_prompt):
        # Truncate prompts to reduce context length
        system_prompt = system_prompt[:200]
        user_prompt = user_prompt[:500]
        
        if "mistral" in self.model.lower():
            prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.generator.device)
            
            with torch.no_grad():
                outputs = self.generator.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("[/INST]")[-1].strip()
            
        elif "phi" in self.model.lower():
            prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.generator.device)
            
            with torch.no_grad():
                outputs = self.generator.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("Assistant:")[-1].strip()
        
        elif "opt" in self.model.lower() or "gptj" in self.model.lower():
            prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.generator.device)
            
            with torch.no_grad():
                outputs = self.generator.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Assistant:")[-1].strip()
            if len(response) > 0:
                words = response.split()
                cleaned_words = []
                prev_word = None
                repeat_count = 0
                for word in words:
                    if word != prev_word:
                        cleaned_words.append(word)
                        repeat_count = 0
                    elif repeat_count < 2:
                        cleaned_words.append(word)
                        repeat_count += 1
                    prev_word = word
                response = ' '.join(cleaned_words)
            return response
        
        elif "tiny-llama" in self.model.lower():
            prompt = f"<|system|>{system_prompt}</s><|user|>{user_prompt}</s><|assistant|>"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.generator.device)
            
            with torch.no_grad():
                outputs = self.generator.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("<|assistant|>")[-1].strip()
            
        elif "falcon" in self.model.lower():
            prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.generator.device)
            
            with torch.no_grad():
                outputs = self.generator.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("Assistant:")[-1].strip()
        
    def chat_gemma(self, system_prompt, user_prompt):
        prompt = f"<bos><start_of_turn>user\nPlease respond to binary questions.\n\n{system_prompt}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model"
        
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            output_ids = self.generator.generate(
                token_ids.to(self.generator.device),
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])

        return str(response)
    


    
