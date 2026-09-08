import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel
from langchain_core.messages import AIMessage


class LocalQwenChatModel:

    BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

    # Your newly trained QLoRA model
    FINETUNED_MODEL = "finetuning/output/qwen-qlora-v2"

    def __init__(self, model_type: str = "finetuned"):

        if model_type not in ["base", "finetuned"]:
            raise ValueError(
                "model_type must be 'base' or 'finetuned'"
            )

        self.model_type = model_type

        print(f"Loading Qwen model: {model_type}")

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is not available."
            )

        print(
            f"Using GPU: {torch.cuda.get_device_name(0)}"
        )

        # 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.BASE_MODEL,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base Qwen model
        print("Loading base Qwen model...")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load QLoRA adapter
        if self.model_type == "finetuned":

            if not os.path.exists(self.FINETUNED_MODEL):
                raise FileNotFoundError(
                    f"Fine-tuned model not found: "
                    f"{self.FINETUNED_MODEL}"
                )

            print(
                f"Loading QLoRA adapter: "
                f"{self.FINETUNED_MODEL}"
            )

            self.model = PeftModel.from_pretrained(
                self.model,
                self.FINETUNED_MODEL,
            )

        self.model.eval()

        print(
            f"Qwen {self.model_type} model loaded successfully."
        )

    # ------------------------------------------------------
    # BUILD CHAT PROMPT
    # ------------------------------------------------------

    def _build_prompt(self, messages):

        if isinstance(messages, str):
            return messages

        formatted_messages = []

        for message in messages:

            if hasattr(message, "content"):

                content = message.content

                if message.__class__.__name__ == "HumanMessage":
                    role = "user"

                elif message.__class__.__name__ == "AIMessage":
                    role = "assistant"

                else:
                    role = "user"

            elif isinstance(message, dict):

                role = message.get(
                    "role",
                    "user"
                )

                content = message.get(
                    "content",
                    ""
                )

            else:

                role = "user"
                content = str(message)

            formatted_messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # ------------------------------------------------------
    # GENERATE ANSWER
    # ------------------------------------------------------

    def _generate(
        self,
        messages,
        max_new_tokens=512,
        temperature=0.7,
    ):

        prompt = self._build_prompt(messages)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            if temperature <= 0:

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            else:

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        input_length = inputs["input_ids"].shape[1]

        generated_tokens = outputs[0][input_length:]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return answer.strip()

    # ------------------------------------------------------
    # INVOKE
    # ------------------------------------------------------

    def invoke(
        self,
        messages,
        max_new_tokens=512,
        temperature=0.7,
        **kwargs,
    ):

        answer = self._generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return AIMessage(
            content=answer
        )

    # ------------------------------------------------------
    # STREAM RESPONSE
    # ------------------------------------------------------

    def stream_response(
        self,
        messages,
        max_new_tokens=512,
        temperature=0.7,
        **kwargs,
    ):

        answer = self._generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Send the answer in small chunks
        words = answer.split(" ")

        for index, word in enumerate(words):

            if index == 0:
                yield word
            else:
                yield " " + word