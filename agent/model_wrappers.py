import torch
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Base LLM Wrapper Class
# ============================================================
class LLMWrapper(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Loading {model_name} on {self.device.upper()} ---")

    @abstractmethod
    def generate_sql(self, question, schema):
        pass

    def _clean_sql(self, text):
        """
        Removes markdown/code artifacts and truncates after the first semicolon.
        """
        # Remove markdown blocks
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.lower().startswith("sql"):
                    text = text[3:]

        # Remove common LLM artifacts
        text = text.replace("<|assistant|>", "").strip()

        # Only keep the first SQL statement
        if ";" in text:
            text = text.split(";")[0] + ";"

        return text.strip()


# ============================================================
# TinyLlama Chat Wrapper
# ============================================================
class TinyLlamaWrapper(LLMWrapper):
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        super().__init__(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def generate_sql(self, question, schema):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful data scientist. "
                    "Your job is to generate valid SQL queries based on the provided schema. "
                    "Output ONLY the SQL query."
                )
            },
            {
                "role": "user",
                "content": f"Schema:\n{schema}\n\nQuestion: {question}"
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict = True
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

        # Decode only the generated part
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        return self._clean_sql(response)


# ============================================================
# GPT-2 SQL Completion Wrapper
# ============================================================
class GPT2Wrapper(LLMWrapper):
    def __init__(self, model_name="openai-community/gpt2"):
        super().__init__(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        # GPT-2 has no pad token → use EOS
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_sql(self, question, schema):
        # Very explicit prompt — GPT-2 needs strong conditioning
        prompt = f"""
You are an AI system that converts English questions into SQL queries.

Database schema:
{schema}

Example:
Question: How many students exist?
SQL: SELECT COUNT(*) FROM students;

Now convert the next question.

Question: {question}
SQL:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract everything after "SQL:"
        if "SQL:" in full_text:
            sql_part = full_text.split("SQL:")[1].strip()
        else:
            sql_part = full_text.strip()

        # Truncate after first semicolon
        if ";" in sql_part:
            sql_part = sql_part.split(";")[0] + ";"

        return self._clean_sql(sql_part)

class Qwen2Wrapper(LLMWrapper):
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        super().__init__(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            load_in_4bit=True,          # 🔥 4-bit quantization
            torch_dtype=torch.float16,  # mixed precision
        ).to(self.device)

        # Qwen expects an EOS token for controlled decoding
        self.eos = self.tokenizer.eos_token_id

    def generate_sql(self, question, schema):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert SQL generator. "
                    "Given a database schema and a natural language question, "
                    "output ONLY the SQL query. Do not explain."
                )
            },
            {
                "role": "user",
                "content": f"Schema:\n{schema}\n\nQuestion: {question}"
            }
        ]

        # Qwen uses ChatML prompt format
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=120,
            do_sample=False,         # deterministic output
            eos_token_id=self.eos,   # stop at EOS
        )

        # Slice off the prompt
        generated = outputs[0][input_ids.shape[-1]:]

        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return self._clean_sql(text)