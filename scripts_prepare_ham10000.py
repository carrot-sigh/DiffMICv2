#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pickle
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default="/root/autodl-tmp/datasets/HAM10000_metadata.csv")
    parser.add_argument("--images", nargs="+", default=[
        "/root/autodl-tmp/datasets/HAM10000_images_part_1",
        "/root/autodl-tmp/datasets/HAM10000_images_part_2",
        "/root/autodl-tmp/datasets/ham10000_images_part_1",
        "/root/autodl-tmp/datasets/ham10000_images_part_2",
    ])
    parser.add_argument("--out-dir", default="dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    random.seed(args.seed)

    label_map = {
        "akiec": 0,
        "bcc": 1,
        "bkl": 2,
        "df": 3,
        "mel": 4,
        "nv": 5,
        "vasc": 6,
    }

    by_class = defaultdict(list)
    missing = 0

    with open(args.meta, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row["image_id"] + ".jpg"
            img_path = None
            for d in args.images:
                p = os.path.join(d, image_name)
                if os.path.exists(p):
                    img_path = p
                    break
            if img_path is None:
                missing += 1
                continue
            dx = row["dx"]
            if dx not in label_map:
                continue
            by_class[dx].append({"img_root": img_path, "label": label_map[dx]})

    train, test = [], []
    for dx, items in by_class.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * args.train_ratio)
        if n > 1:
            n_train = max(1, min(n - 1, n_train))
        train.extend(items[:n_train])
        test.extend(items[n_train:])

    random.shuffle(train)
    random.shuffle(test)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "ham10000_train_list.pkl"), "wb") as f:
        pickle.dump(train, f)
    with open(os.path.join(args.out_dir, "ham10000_test_list.pkl"), "wb") as f:
        pickle.dump(test, f)
    with open(os.path.join(args.out_dir, "ham10000_label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print("missing_images", missing)
    print("train_size", len(train), "test_size", len(test))


if __name__ == "__main__":
    main()
