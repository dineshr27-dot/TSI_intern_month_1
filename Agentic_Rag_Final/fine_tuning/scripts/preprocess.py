from datasets import load_from_disk

# Load our locally saved dataset
dataset = load_from_disk(
    "finetuning/data/train_dataset"
)

print("Number of examples:", len(dataset))

# Use 1,000 examples for first experiment
dataset = dataset.select(range(1000))


def format_example(example):
    return {
        "text": f"""### Instruction:
{example["instruction"]}

### Response:
{example["response"]}"""
    }


dataset = dataset.map(format_example)

print("\nFormatted example:")
print(dataset[0]["text"])

#save the dataset to disk

dataset.save_to_disk(
    "finetuning/data/processed_dataset"
)

print("\nProcessed dataset saved!")