import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import LoraConfig

from trl import SFTTrainer, SFTConfig


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

OUTPUT_DIR = "./tinyllama-lora"

NUM_SAMPLES = 100


# ============================================================
# 1. CHECK GPU
# ============================================================

print("=" * 60)
print("GPU CHECK")
print("=" * 60)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
else:
    raise RuntimeError("CUDA GPU not available!")


# ============================================================
# 2. LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 60)
print("LOADING TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded!")


# ============================================================
# 3. 4-BIT QLoRA CONFIGURATION
# ============================================================

print("\n" + "=" * 60)
print("CREATING 4-BIT CONFIGURATION")
print("=" * 60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True,
)

print("4-bit configuration created!")


# ============================================================
# 4. LOAD TINYLLAMA IN 4-BIT
# ============================================================

print("\n" + "=" * 60)
print("LOADING TINYLLAMA IN 4-BIT")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)

model.config.use_cache = False

print("Model loaded!")

print("Model:", MODEL_NAME)

print(
    "Parameters:",
    model.num_parameters()
)


# ============================================================
# 5. LOAD ALPACA DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

dataset = load_dataset(
    "yahma/alpaca-cleaned",
    split="train"
)

print("Full dataset loaded!")

print("Total samples:", len(dataset))


# ============================================================
# 6. SELECT SMALL DATASET
# ============================================================

dataset = dataset.select(
    range(min(NUM_SAMPLES, len(dataset)))
)

print("Training samples:", len(dataset))


# ============================================================
# 7. FORMAT DATASET
# ============================================================

print("\nFormatting dataset...")


def format_prompt(example):

    instruction = example["instruction"]

    input_text = example["input"]

    output = example["output"]

    if input_text and input_text.strip():

        text = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""

    else:

        text = f"""### Instruction:
{instruction}

### Response:
{output}"""

    return {
        "text": text
    }


dataset = dataset.map(
    format_prompt
)

print("Dataset formatted!")

print("\nFirst training example:")
print("-" * 60)

print(dataset[0]["text"])

print("-" * 60)


# ============================================================
# 8. LoRA CONFIGURATION
# ============================================================

print("\n" + "=" * 60)
print("CREATING LoRA CONFIGURATION")
print("=" * 60)

lora_config = LoraConfig(
    r=8,

    lora_alpha=16,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],

    bias="none",

    task_type="CAUSAL_LM",
)

print("LoRA configuration created!")


# ============================================================
# 9. SFT TRAINING CONFIGURATION
# ============================================================

print("\n" + "=" * 60)
print("CREATING SFT CONFIGURATION")
print("=" * 60)

sft_config = SFTConfig(

    output_dir=OUTPUT_DIR,

    # Small batch because RTX 4050 has 6 GB VRAM
    per_device_train_batch_size=1,

    # Simulates batch size 4
    gradient_accumulation_steps=4,

    # First test: only 1 epoch
    num_train_epochs=1,

    # LoRA learning rate
    learning_rate=2e-4,

    # Use FP16 on RTX 4050
    fp16=True,

    # Log every step
    logging_steps=1,

    # Save checkpoints
    save_strategy="steps",

    save_steps=20,

    save_total_limit=1,

    # Disable external logging
    report_to="none",

    # Memory-efficient optimizer
    optim="paged_adamw_8bit",

    # Reduce GPU memory usage
    gradient_checkpointing=True,

    # Dataset text column
    dataset_text_field="text",

    # Maximum sequence length
    max_length=256,

)

print("SFT configuration created!")


# ============================================================
# 10. CREATE SFT TRAINER
# ============================================================

print("\n" + "=" * 60)
print("CREATING SFT TRAINER")
print("=" * 60)

trainer = SFTTrainer(

    model=model,

    args=sft_config,

    train_dataset=dataset,

    processing_class=tokenizer,

    peft_config=lora_config,

)

print("Trainer created successfully!")


# ============================================================
# 11. SHOW TRAINABLE PARAMETERS
# ============================================================

print("\n" + "=" * 60)
print("TRAINABLE PARAMETERS")
print("=" * 60)

trainer.model.print_trainable_parameters()


# ============================================================
# 12. START TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING QLoRA TRAINING")
print("=" * 60)

print("Samples:", len(dataset))

print("Epochs:", 1)

print("Batch size:", 1)

print("Gradient accumulation:", 4)

print("Max length:", 256)

print("\nTraining started...\n")


train_result = trainer.train()


# ============================================================
# 13. TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("\nTraining metrics:")

print(train_result.metrics)


# ============================================================
# 14. SAVE LoRA ADAPTER
# ============================================================

print("\n" + "=" * 60)
print("SAVING LoRA ADAPTER")
print("=" * 60)

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print("\nLoRA adapter saved successfully!")

print("Location:")

print(OUTPUT_DIR)


# ============================================================
# 15. FINAL GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    allocated = torch.cuda.memory_allocated() / 1024**3

    reserved = torch.cuda.memory_reserved() / 1024**3

    print("\nGPU memory allocated:",
          round(allocated, 2), "GB")

    print("GPU memory reserved:",
          round(reserved, 2), "GB")


print("\n" + "=" * 60)
print("QLoRA PIPELINE FINISHED")
print("=" * 60)