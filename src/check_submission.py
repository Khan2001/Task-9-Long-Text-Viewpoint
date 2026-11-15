"""提交前格式校验：复用官方口径的 metric.validate，检查 test 预测文件是否可直接提交。

用法:
    .venv/bin/python src/check_submission.py --pred preds/test_locked.jsonl
"""

import argparse
import codecs
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metric import validate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "test.jsonl"))
    args = ap.parse_args()

    rows = read_jsonl(args.pred)
    samples = read_jsonl(args.data)
    texts = {s["sample_id"]: s["docs"][0]["full_text"] for s in samples}
    ids = [s["sample_id"] for s in samples]

    raw = open(args.pred, "rb").read()
    errors = validate(rows, gold_ids=ids, texts=texts)

    stances, n_issue, empty_future = {}, 0, 0
    for r in rows:
        for it in r.get("issue_list") or []:
            n_issue += 1
            stances[it.get("stance")] = stances.get(it.get("stance"), 0) + 1
        if not any(str(x).strip() for x in (r.get("future_argument") or [])):
            empty_future += 1

    print(f"文件: {args.pred}")
    bom = raw[:3] == codecs.BOM_UTF8
    print(f"字节: {len(raw)}  | BOM: {bom}  | 行数: {len(rows)} / 期望 {len(ids)}")
    print(f"议题总数: {n_issue}  (平均每篇 {n_issue / max(1, len(rows)):.2f})")
    print(f"立场分布: {stances}")
    print(f"future 全空篇数: {empty_future}")
    print(f"validate 报错数: {len(errors)}")
    for e in errors[:20]:
        print(f"  ! {e}")
    if len(errors) > 20:
        print(f"  ... 其余 {len(errors) - 20} 条省略")
    print("\n结论: " + ("可以提交" if not errors else "有问题，先别覆盖 result.jsonl"))


if __name__ == "__main__":
    main()
