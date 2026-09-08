from datasets import load_dataset


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_NAME = "VMware/open-instruct-v1-oasst-dolly-hhrlhf"

TRAIN_PATH = "finetuning/data/train_dataset"
TEST_PATH = "finetuning/data/test_dataset"

TEST_SIZE = 0.20
RANDOM_SEED = 42


# ============================================================
# 2. LOAD ORIGINAL DATASET
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

dataset = load_dataset(DATASET_NAME)

print(dataset)


# ============================================================
# 3. GET TRAIN SPLIT
# ============================================================

full_dataset = dataset["train"]

print()
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Total examples:", len(full_dataset))


# ============================================================
# 4. SHOW FIRST EXAMPLE
# ============================================================

print()
print("FIRST EXAMPLE:")
print(full_dataset[0])


# ============================================================
# 5. SPLIT DATASET 80% / 20%
# ============================================================

print()
print("=" * 60)
print("CREATING 80% / 20% SPLIT")
print("=" * 60)

split_dataset = full_dataset.train_test_split(
    test_size=TEST_SIZE,
    seed=RANDOM_SEED,
)

train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]


# ============================================================
# 6. SHOW SPLIT INFORMATION
# ============================================================

print()
print("=" * 60)
print("DATASET SPLIT RESULT")
print("=" * 60)

print("Total examples :", len(full_dataset))
print("Training 80%   :", len(train_dataset))
print("Testing 20%    :", len(test_dataset))


# ============================================================
# 7. SAVE TRAINING DATASET
# ============================================================

print()
print("=" * 60)
print("SAVING TRAINING DATASET")
print("=" * 60)

train_dataset.save_to_disk(TRAIN_PATH)

print("Training dataset saved to:")
print(TRAIN_PATH)


# ============================================================
# 8. SAVE TESTING DATASET
# ============================================================

print()
print("=" * 60)
print("SAVING TESTING DATASET")
print("=" * 60)

test_dataset.save_to_disk(TEST_PATH)

print("Testing dataset saved to:")
print(TEST_PATH)


# ============================================================
# 9. SHOW TRAINING EXAMPLE
# ============================================================

print()
print("=" * 60)
print("TRAINING EXAMPLE")
print("=" * 60)

print(train_dataset[0])


# ============================================================
# 10. SHOW TESTING EXAMPLE
# ============================================================

print()
print("=" * 60)
print("TESTING EXAMPLE")
print("=" * 60)

print(test_dataset[0])


# ============================================================
# 11. FINAL RESULT
# ============================================================

print()
print("=" * 60)
print("DATASET PREPARATION COMPLETED")
print("=" * 60)

print("Total examples :", len(full_dataset))
print("Training data  :", len(train_dataset), "(80%)")
print("Testing data   :", len(test_dataset), "(20%)")

print()
print("Training dataset:")
print(TRAIN_PATH)

print()
print("Testing dataset:")
print(TEST_PATH)

print()
print("Dataset split and saved successfully!")