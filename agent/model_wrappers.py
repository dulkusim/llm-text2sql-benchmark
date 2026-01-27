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

    def _clean_sql(self, text: str) -> str:
        """
        Removes markdown/code artifacts and truncates after the first semicolon.
        """
        if not text:
            return ""

        # Remove markdown blocks
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.lower().lstrip().startswith("sql"):
                    text = text.split("\n", 1)[-1]

        # Remove common artifacts
        text = text.replace("<|assistant|>", "").strip()

        # If model returns some preface, try to grab after "SQL:"
        if "SQL:" in text:
            text = text.split("SQL:", 1)[1].strip()

        # Keep only first statement
        if ";" in text:
            text = text.split(";")[0].strip() + ";"

        return text.strip()


# ============================================================
# TinyLlama Wrapper (Completion-style, reliable for CSV SQL)
# ============================================================
class TinyLlamaWrapper(LLMWrapper):
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        super().__init__(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Some LLaMA-like tokenizers may not have pad token set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def generate_sql(self, question, schema):
        # Completion-style prompt works more reliably than chat-template for TinyLlama
        prompt = f"""You are an expert Text-to-SQL system.

Rules:
- Use ONLY tables and columns from the schema.
- Output ONLY the SQL query (no explanation, no markdown).

Schema:
{schema}

Question:
{question}

SQL:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract after "SQL:" if present
        if "SQL:" in full_text:
            sql_part = full_text.split("SQL:", 1)[1].strip()
        else:
            sql_part = full_text.strip()

        return self._clean_sql(sql_part)


# ============================================================
# Qwen 2.5 Wrapper (Colab-safe, stable)
# ============================================================
class Qwen2Wrapper(LLMWrapper):
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        super().__init__(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Qwen tokenizer usually has eos/pad, but keep safe fallback
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        )

        self.eos = self.tokenizer.eos_token_id

    def generate_sql(self, question, schema):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Text-to-SQL system. "
                    "Use ONLY the tables and columns provided in the schema. "
                    "Do NOT hallucinate tables or columns. "
                    "Output ONLY a valid SQL query."
                )
            },
            {
                "role": "user",
                "content": f"Schema:\n{schema}\n\nQuestion: {question}"
            }
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=64,
                do_sample=False,
                eos_token_id=self.eos,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = outputs[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return self._clean_sql(text)