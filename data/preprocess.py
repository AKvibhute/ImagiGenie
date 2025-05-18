import os
from datasets import load_dataset

# ✅ Parameters
MAX_STORIES = 1000  # Keep small due to limited GPU memory
DELIMITER = "<|endofstory|>"

# ✅ Load TinyStories dataset from Hugging Face
dataset = load_dataset("roneneldan/TinyStories", split="train")

# ✅ Output path
output_file = "data/processed/stories.txt"
os.makedirs("data/processed", exist_ok=True)

# ✅ Write cleaned and trimmed stories
with open(output_file, 'w', encoding='utf-8') as out:
    count = 0
    for i, example in enumerate(dataset):
        if count >= MAX_STORIES:
            break

        story = example.get("text", "").strip()
        if not story:
            continue

        # ✅ Basic cleanup
        story = story.replace("\n", " ").replace("  ", " ").strip()

        # ✅ Skip very short or long stories
        if len(story.split()) < 30 or len(story.split()) > 250:
            continue

        # ✅ Write formatted story with delimiter
        out.write(f"{story} {DELIMITER}\n\n")
        count += 1

print(f"✅ Preprocessed {count} stories and saved to {output_file}")
