"""Дополнительные проверки; выданный check.sh остаётся неизменным."""

import copy
import hashlib
import json
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

from src.collate import DynamicPaddingCollator
from src.config import load_params
from src.prompt import build_chat_text, prompt_token_len
from src.tokenize_data import mask_prompt, process_split, read_jsonl, truncation_stats


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = load_params()
        cls.tok = AutoTokenizer.from_pretrained(cls.params["model"]["name"])
        cls.record = {
            "id": "synthetic", "topic": "test",
            "messages": [
                {"role": "system", "content": "Отвечай кратко."},
                {"role": "user", "content": "Как сложить два числа?"},
                {"role": "assistant", "content": "Используй a + b."},
            ],
        }

    def test_input_snapshot_and_attribution(self):
        manifest = json.loads(Path("docs/input_manifest.json").read_text(encoding="utf-8"))
        for name, meta in manifest["files"].items():
            path = Path(self.params["data"]["train_jsonl"]).parent / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), meta["sha256"])
            if "records" in meta:
                self.assertEqual(len(read_jsonl(path)), meta["records"])
        attribution = {r["id"] for r in read_jsonl(path.parent / "attribution.jsonl")}
        for key in ("train_jsonl", "val_jsonl"):
            self.assertTrue({r["id"] for r in read_jsonl(Path(self.params["data"][key]))} <= attribution)

    def test_all_saved_answers_both_splits(self):
        for name, key in (("train", "train_jsonl"), ("val", "val_jsonl")):
            records = {r["id"]: r for r in read_jsonl(Path(self.params["data"][key]))}
            blob = torch.load(Path(self.params["data"]["out_dir"]) / f"{name}.pt", weights_only=True)
            for ex in blob["examples"]:
                with self.subTest(split=name, id=ex["id"]):
                    record = records[ex["id"]]
                    train = build_chat_text(self.tok, record["messages"], self.params, False)
                    prompt = build_chat_text(self.tok, record["messages"], self.params, True)
                    self.assertTrue(train.encode().startswith(prompt.encode()))
                    supervised = [token for token, label in zip(ex["input_ids"], ex["labels"]) if label != -100]
                    self.assertTrue(supervised)
                    got = self.tok.decode(supervised, skip_special_tokens=True).strip()
                    want = record["messages"][-1]["content"].strip()
                    if len(ex["input_ids"]) >= self.params["tokenize"]["max_seq_len"]:
                        self.assertTrue(want.startswith(got))
                    else:
                        self.assertEqual(got, want)
                    self.assertTrue(all(label == -100 or label == token for token, label in zip(ex["input_ids"], ex["labels"])))

    def test_prompt_without_assistant_is_supported(self):
        messages = self.record["messages"]
        self.assertEqual(build_chat_text(self.tok, messages, self.params, True),
                         build_chat_text(self.tok, messages[:-1], self.params, True))

    def test_mask_and_answer_fully_truncated(self):
        self.assertEqual(mask_prompt([1, 2, 3], 2), [-100, -100, 3])
        self.assertEqual(mask_prompt([1, 2], 8), [-100, -100])
        params = copy.deepcopy(self.params)
        params["tokenize"]["max_seq_len"] = 1
        with patch("src.tokenize_data.read_jsonl", return_value=[self.record]), warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            examples, stats = process_split(self.tok, "synthetic", Path("unused"), params)
        self.assertEqual(examples, [])
        self.assertEqual(stats["dropped_no_supervision"], 1)
        self.assertEqual(stats["length_tokens"]["count"], 1)
        self.assertEqual(stats["truncated_ratio"], 1.0)
        self.assertTrue(emitted)

    def test_warning_above_threshold_not_at_threshold(self):
        params = copy.deepcopy(self.params)
        params["tokenize"]["truncated_warn_ratio"] = 0.5
        metas = [{"truncated": True, "full_len": 4, "answer_len": 2, "supervised": 1},
                 {"truncated": False, "full_len": 2, "answer_len": 1, "supervised": 1}]
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            self.assertEqual(truncation_stats(metas, "test", params)["truncated_ratio"], 0.5)
            self.assertFalse(emitted)
            truncation_stats(metas[:1], "test", params)
            self.assertEqual(len(emitted), 1)

    def test_bpe_boundary_crossing_token_is_masked(self):
        prompt, full = "Приве", "Привет, мир"
        encoded = self.tok(full, add_special_tokens=False, return_offsets_mapping=True)
        n, fallback = prompt_token_len(self.tok, prompt, encoded["input_ids"], encoded["offset_mapping"])
        self.assertTrue(fallback)
        labels = mask_prompt(encoded["input_ids"], n)
        for i, (start, end) in enumerate(encoded["offset_mapping"]):
            if start < len(prompt):
                self.assertEqual(labels[i], -100)
        self.assertGreaterEqual(encoded["offset_mapping"][n][0], len(prompt))

    def test_dynamic_padding_values_and_validation(self):
        collator = DynamicPaddingCollator(self.tok.pad_token_id)
        examples = [{"input_ids": [1], "attention_mask": [1], "labels": [1]},
                    {"input_ids": [2, 3, 4], "attention_mask": [1, 1, 1], "labels": [-100, 3, 4]}]
        batch = collator(examples)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 3))
        self.assertEqual(batch["attention_mask"][0].tolist(), [0, 0, 1])
        self.assertEqual(batch["labels"][0].tolist(), [-100, -100, 1])
        self.assertEqual(batch["input_ids"][0].tolist(), [self.tok.pad_token_id, self.tok.pad_token_id, 1])
        with self.assertRaises(ValueError):
            collator([])
        with self.assertRaises(ValueError):
            collator([{"input_ids": [1], "attention_mask": [], "labels": [1]}])


if __name__ == "__main__":
    unittest.main()
