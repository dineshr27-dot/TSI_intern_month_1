from datasets import load_dataset

dataset = load_dataset(
    "VMware/open-instruct-v1-oasst-dolly-hhrlhf"
)

print(dataset)

print("\nFirst example:")
print(dataset["train"][0])

print("\nNumber of examples:")
print(len(dataset["train"]))

# Save training dataset locally
dataset["train"].save_to_disk(
    "finetuning/data/train_dataset"
)

print("\nDataset saved successfully!")