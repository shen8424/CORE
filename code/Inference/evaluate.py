"""
评估通用二分类推理结果：计算整体 Accuracy、Precision、Recall、F1，以及按 fake_cls 分类的 Accuracy。

Usage:
    python evaluate.py [--result_path PATH]
"""
import json
import re
import argparse
from collections import defaultdict


def clean(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text).strip().lower()


def evaluate(result_path: str):
    y_true, y_pred = [], []
    cls_stats = defaultdict(lambda: [0, 0])  # [correct, total]
    invalid = 0

    with open(result_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            label = clean(item['labels'])
            response = clean(item['response'])
            cls = item.get('fake_cls', 'unknown')

            if label in ['real', 'fake'] and response in ['real', 'fake']:
                t = 1 if label == 'real' else 0
                p = 1 if response == 'real' else 0
                y_true.append(t)
                y_pred.append(p)
                cls_stats[cls][1] += 1
                if t == p:
                    cls_stats[cls][0] += 1
            else:
                invalid += 1

    total = len(y_true)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    acc = correct / total if total else 0

    # Precision / Recall / F1 (for 'real' class, pos_label=1)
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("=" * 60)
    print(f"  Results: {result_path}")
    print("=" * 60)
    print(f"  Total: {total}  |  Correct: {correct}  |  Invalid: {invalid}")
    print("-" * 60)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}  (Real class)")
    print(f"  Recall:    {recall:.4f}  (Real class)")
    print(f"  F1 Score:  {f1:.4f}  (Real class)")
    print("-" * 60)
    print(f"  {'fake_cls':<30} {'Total':>6} {'Correct':>8} {'Acc':>8}")
    print("-" * 60)
    for cls, (c, t) in sorted(cls_stats.items()):
        print(f"  {cls:<30} {t:>6} {c:>8} {c/t:>8.4f}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str,
                        default='./Outputs/inference/stage3_vllm_results.jsonl')
    args = parser.parse_args()
    evaluate(args.result_path)
