"""
One-time mechanical pass: computes length_bucket and sets data_source on every
train/test example (idempotent — never overwrites an existing data_source, so
future real-data ingestion stays correctly tagged), then rebuilds
data/dataset_manifest.json from data/test/*.json.

Run after any new train/test JSON files are added:
    python scripts/build_dataset_manifest.py
"""
import json
import os


def word_count(text: str) -> int:
    return len(text.split())


def length_bucket(wc: int) -> str:
    if wc < 40:
        return "short"
    if wc <= 100:
        return "medium"
    return "long"


def annotate(split: str):
    for fname in sorted(os.listdir(f"data/{split}")):
        if not fname.endswith(".json"):
            continue
        path = f"data/{split}/{fname}"
        with open(path) as f:
            data = json.load(f)
        data["length_bucket"] = length_bucket(word_count(data["answer_text"]))
        data.setdefault("data_source", "synthetic")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def rebuild_manifest():
    manifest = []
    for fname in sorted(os.listdir("data/test")):
        if not fname.endswith(".json"):
            continue
        with open(f"data/test/{fname}") as f:
            d = json.load(f)
        manifest.append({
            "test_id": d["answer_id"],
            "derived_from_train_id": d["derived_from_train_id"],
            "question_id": d["question_id"],
            "variant_type": d["variant_type"],
            "length_bucket": d["length_bucket"],
            "data_source": d["data_source"],
        })
    with open("data/dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    annotate("train")
    annotate("test")
    rebuild_manifest()
    print("length_bucket + data_source annotated on all train/test files; dataset_manifest.json rebuilt.")


if __name__ == "__main__":
    main()
