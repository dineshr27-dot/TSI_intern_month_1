import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("=" * 50)
print("Loading tokenizer...")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded!")

print("\nLoading TinyLlama...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("\n" + "=" * 50)
print("MODEL LOADED SUCCESSFULLY")
print("=" * 50)

print("Model:", MODEL_NAME)
print("GPU:", torch.cuda.get_device_name(0))
print("Device:", model.device)
print("Parameters:", sum(p.numel() for p in model.parameters()))

question = "What is machine learning? Explain in simple words."

prompt = f"""<|system|>
You are a helpful AI assistant.
<|user|>
{question}
<|assistant|>
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

answer = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print("\nQuestion:", question)
print("\nAnswer:", answer)