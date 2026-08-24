from datasets import load_dataset

print("=" * 50)
print("Loading Alpaca dataset from Hugging Face...")
print("=" * 50)

dataset = load_dataset(
    "yahma/alpaca-cleaned",
    split="train"
)

print("\nFull dataset loaded!")
print("Total samples:", len(dataset))

# Take only 100 samples for our first local experiment
dataset = dataset.select(range(100))

print("Samples selected:", len(dataset))

print("\nDataset columns:")
print(dataset.column_names)


def format_prompt(example):

    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]

    if input_text.strip():

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

    return {"text": text}


dataset = dataset.map(format_prompt)

print("\n" + "=" * 50)
print("DATASET READY")
print("=" * 50)

print("Number of samples:", len(dataset))

print("\nFirst formatted sample:")
print(dataset[0]["text"])