import torch

from datasets import load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
)

from trl import SFTTrainer


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# 80% TRAINING DATASET
DATASET_PATH = "finetuning/data/train_dataset"

# NEW MODEL OUTPUT
# Existing qwen-qlora will NOT be changed
OUTPUT_DIR = "finetuning/output/qwen-qlora-v2"


# ============================================================
# 2. CHECK GPU
# ============================================================

print("=" * 60)
print("GPU INFORMATION")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available.")

print("GPU:", torch.cuda.get_device_name(0))
print()


# ============================================================
# 3. LOAD TRAINING DATASET
# ============================================================

print("=" * 60)
print("LOADING TRAINING DATASET")
print("=" * 60)

dataset = load_from_disk(DATASET_PATH)

print("Training examples:", len(dataset))
print("Dataset columns:", dataset.column_names)
print()


# ============================================================
# 4. FORMAT DATASET FOR SFT TRAINING
# ============================================================

print("=" * 60)
print("FORMATTING TRAINING DATASET")
print("=" * 60)


def format_example(example):
    return {
        "text": (
            "### Instruction:\n"
            + str(example["instruction"])
            + "\n\n"
            "### Response:\n"
            + str(example["response"])
        )
    }


dataset = dataset.map(
    format_example,
    remove_columns=dataset.column_names,
)


print("Dataset formatted successfully.")
print("Final dataset columns:", dataset.column_names)

print("\nFirst formatted example:")
print(dataset[0]["text"])

print()


# ============================================================
# 5. LOAD TOKENIZER
# ============================================================

print("=" * 60)
print("LOADING TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded successfully")
print()


# ============================================================
# 6. 4-BIT QUANTIZATION
# ============================================================

print("=" * 60)
print("SETTING UP 4-BIT QUANTIZATION")
print("=" * 60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    # NF4 quantization
    bnb_4bit_quant_type="nf4",

    # FP16 computation
    bnb_4bit_compute_dtype=torch.float16,

    # Double quantization
    bnb_4bit_use_double_quant=True,
)

print("4-bit NF4 quantization enabled")
print()


# ============================================================
# 7. LOAD BASE MODEL
# ============================================================

print("=" * 60)
print("LOADING QWEN MODEL")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# Disable cache during training
model.config.use_cache = False

print("Model loaded successfully")
print()


# ============================================================
# 8. PREPARE MODEL FOR QLORA
# ============================================================

print("=" * 60)
print("PREPARING MODEL FOR QLORA")
print("=" * 60)

model = prepare_model_for_kbit_training(model)

print("Model prepared for k-bit training")
print()


# ============================================================
# 9. LoRA CONFIGURATION
# ============================================================

print("=" * 60)
print("SETTING UP LoRA")
print("=" * 60)

lora_config = LoraConfig(
    # LoRA rank
    r=8,

    # LoRA scaling
    lora_alpha=16,

    # LoRA dropout
    lora_dropout=0.05,

    # Do not train bias
    bias="none",

    # Causal language model
    task_type="CAUSAL_LM",

    # Qwen transformer modules
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

print("LoRA configuration created")
print()


# ============================================================
# 10. TRAINING SETTINGS
# ============================================================

print("=" * 60)
print("SETTING UP TRAINING")
print("=" * 60)

training_args = TrainingArguments(

    # New output directory
    output_dir=OUTPUT_DIR,

    # One complete pass through training dataset
    num_train_epochs=1,

    # One example per GPU step
    per_device_train_batch_size=1,

    # Accumulate gradients
    gradient_accumulation_steps=8,

    # LoRA learning rate
    learning_rate=2e-4,

    # Disable AMP
    fp16=False,
    bf16=False,

    # Show loss every 10 steps
    logging_steps=10,

    # Save checkpoint every 100 steps
    save_steps=100,

    # Keep only two checkpoints
    save_total_limit=2,

    # Memory-efficient optimizer
    optim="paged_adamw_8bit",

    # No external reporting
    report_to="none",

    # Save GPU memory
    gradient_checkpointing=True,

    # Disable gradient clipping
    max_grad_norm=0.0,
)

print("Training configuration created")
print()

print("Epochs:", training_args.num_train_epochs)
print("Batch size:", training_args.per_device_train_batch_size)
print("Gradient accumulation:", training_args.gradient_accumulation_steps)
print("Learning rate:", training_args.learning_rate)
print("FP16:", training_args.fp16)
print("BF16:", training_args.bf16)
print("Max grad norm:", training_args.max_grad_norm)
print()


# ============================================================
# 11. CREATE SFT TRAINER
# ============================================================

print("=" * 60)
print("CREATING SFT TRAINER")
print("=" * 60)

trainer = SFTTrainer(
    model=model,

    args=training_args,

    train_dataset=dataset,

    peft_config=lora_config,
)

print("SFTTrainer created")
print()


# ============================================================
# 12. START TRAINING
# ============================================================

print("=" * 60)
print("STARTING QLoRA TRAINING")
print("=" * 60)

print()
print("Training dataset:", DATASET_PATH)
print("Training examples:", len(dataset))
print("Output directory:", OUTPUT_DIR)
print()

trainer.train()


# ============================================================
# 13. SAVE TRAINED LoRA ADAPTER
# ============================================================

print()
print("=" * 60)
print("SAVING TRAINED MODEL")
print("=" * 60)

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print()
print("Training completed successfully!")
print(f"Model saved to: {OUTPUT_DIR}")
print()