# coding=utf-8
"""Small persistence helpers used by the training entry points."""

import json
import os
import pickle


def save_json_data(output_dir, filename, data):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


def save_pickle_data(output_dir, filename, data):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(data, f)
