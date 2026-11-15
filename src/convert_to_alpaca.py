"""把赛题 train.jsonl 转成单轮 Alpaca 指令微调数据。

用法:
    python src/convert_to_alpaca.py \
        --input data/train.jsonl \
        --output data/train_alpaca.jsonl
"""

import argparse
import json
import os
import sys


SYSTEM_PROMPT = (
    "你是一名谈判情报分析师。你需要从长文本中识别主议题与子议题，判断立场，"
    "抽取原文证据链，并预测各议题后续可能的观点与论证方向。"
    "只输出合法 JSON，不输出解释文字或 Markdown 代码围栏。"
)

INSTRUCTION = (
    "分析输入文档，完成以下任务：\n"
    "1. 识别主议题和相关子议题，并按主议题在前、子议题在后的顺序输出；\n"
    "2. 为每个议题判断立场，stance 只能取 support、oppose、neutral；\n"
    "3. 为每个议题抽取 1-3 段支撑立场的原文证据，argument_chain 中的内容必须是"
    "输入文档中的逐字片段，不得改写或编造；\n"
    "4. 预测每个议题后续可能出现的观点和论证方向，future_argument 必须与 "
    "issue_list 数量相同且逐项同序对应，每条 60-110 字；\n"
    "5. 仅输出以下 JSON，不要输出 sample_id：\n"
    "{\"issue_list\":[{\"issue_name\":\"...\",\"stance\":\"support\","
    "\"argument_chain\":[\"...\"]}],\"future_argument\":[\"...\"]}"
)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON 解析失败: {exc}") from exc
    return rows


def format_documents(sample):
    docs = sample.get("docs")
    if not isinstance(docs, list) or not docs:
        raise ValueError(f"{sample.get('sample_id', '<unknown>')}: docs 为空")

    parts = ["以下文档按时间顺序排列："]
    for index, doc in enumerate(docs, 1):
        if not isinstance(doc, dict):
            raise ValueError(f"{sample.get('sample_id', '<unknown>')}: docs[{index - 1}] 不是对象")
        text = str(doc.get("full_text") or "").strip()
        if not text:
            raise ValueError(f"{sample.get('sample_id', '<unknown>')}: docs[{index - 1}].full_text 为空")
        parts.append(
            f"<<<DOC {index} doc_id={doc.get('doc_id', '')} "
            f"doc_type={doc.get('doc_type', '')} publish_date={doc.get('publish_date', '')}>>>"
        )
        parts.append(text)
        parts.append("<<<END_DOC>>>")
    return "\n".join(parts)


def build_output(sample):
    issues = sample.get("issue_list")
    future = sample.get("future_argument")
    sample_id = sample.get("sample_id", "<unknown>")

    if not isinstance(issues, list) or not issues:
        raise ValueError(f"{sample_id}: issue_list 为空")
    if not isinstance(future, list):
        raise ValueError(f"{sample_id}: future_argument 不是列表")
    if len(issues) != len(future):
        raise ValueError(
            f"{sample_id}: issue_list 数量 {len(issues)} 与 "
            f"future_argument 数量 {len(future)} 不一致"
        )

    normalized_issues = []
    for index, issue in enumerate(issues):
        if not isinstance(issue, dict):
            raise ValueError(f"{sample_id}: issue_list[{index}] 不是对象")

        name = str(issue.get("issue_name") or "").strip()
        stance = str(issue.get("stance") or "").strip()
        chain = issue.get("argument_chain")

        if not name:
            raise ValueError(f"{sample_id}: issue_list[{index}].issue_name 为空")
        if stance not in {"support", "oppose", "neutral"}:
            raise ValueError(f"{sample_id}: issue_list[{index}].stance 非法: {stance!r}")
        if not isinstance(chain, list) or not chain:
            raise ValueError(f"{sample_id}: issue_list[{index}].argument_chain 为空")

        normalized_chain = [str(item).strip() for item in chain if str(item).strip()]
        if not normalized_chain:
            raise ValueError(f"{sample_id}: issue_list[{index}].argument_chain 无有效内容")

        normalized_issues.append({
            "issue_name": name,
            "stance": stance,
            "argument_chain": normalized_chain,
        })

    normalized_future = [str(item).strip() for item in future if str(item).strip()]
    if len(normalized_future) != len(normalized_issues):
        raise ValueError(f"{sample_id}: future_argument 存在空内容")

    return {
        "issue_list": normalized_issues,
        "future_argument": normalized_future,
    }


def convert_sample(sample):
    target = build_output(sample)
    return {
        "system": SYSTEM_PROMPT,
        "instruction": INSTRUCTION,
        "input": format_documents(sample),
        "output": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
        "history": [],
    }


def write_jsonl(path, rows):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="赛题数据转 Alpaca JSONL")
    parser.add_argument("--input", default="data/train.jsonl", help="原始 JSONL 路径")
    parser.add_argument("--output", default="data/train_alpaca.jsonl", help="输出 JSONL 路径")
    parser.add_argument("--limit", type=int, help="仅转换前 N 条，用于快速检查")
    args = parser.parse_args()

    samples = read_jsonl(args.input)
    if args.limit is not None:
        samples = samples[:args.limit]
    if not samples:
        sys.exit(f"没有可转换样本: {args.input}")

    rows = [convert_sample(sample) for sample in samples]
    write_jsonl(args.output, rows)

    input_chars = sum(len(row["input"]) for row in rows)
    output_chars = sum(len(row["output"]) for row in rows)
    print(
        f"转换完成: {len(rows)} 条 | 输出 {args.output} | "
        f"平均 input {input_chars / len(rows):.0f} 字符 | "
        f"平均 output {output_chars / len(rows):.0f} 字符"
    )


if __name__ == "__main__":
    main()
