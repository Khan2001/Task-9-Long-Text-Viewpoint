import json
from pathlib import Path

DATA_DIR = Path("/workspace/qwen3-finetune/data")

INSTRUCTION = """你是一名专业的事件分析与舆情研判助手。

请根据提供的新闻材料，完成以下任务：

1. 识别新闻材料中的核心议题及重要子议题。
2. 为每个议题生成简洁、准确的 issue_name。
3. 判断每个议题的立场 stance，只能使用 support、oppose、neutral。
4. 从原始材料中提取支持该议题的关键论据，组成 argument_chain。
5. 根据事件发展趋势，为各议题预测未来可能出现的观点，形成 future_argument。

要求：
- issue_list 按重要程度排序，核心议题放在第一位。
- issue_name 使用简洁的中文名词短语。
- argument_chain 尽量使用新闻原文，不得编造事实。
- future_argument 应结合现有事实，预测合理的未来发展观点。
- 输出必须是合法 JSON。
- 不要输出 Markdown 代码块。
- 不要输出解释、分析过程或额外文字。

输出格式：
{
  "issue_list": [
    {
      "issue_name": "议题名称",
      "stance": "support",
      "argument_chain": [
        "论据1",
        "论据2"
      ]
    }
  ],
  "future_argument": [
    "未来论点1"
  ]
}"""


def build_input(docs):
    """
    将一条样本中的多篇新闻组织成统一输入。
    """

    parts = []

    for i, doc in enumerate(docs, start=1):

        doc_id = doc.get("doc_id", "")
        doc_type = doc.get("doc_type", "")
        publish_date = doc.get("publish_date", "")
        full_text = doc.get("full_text", "")

        text = (
            f"【新闻{i}】\n"
            f"文档编号：{doc_id}\n"
            f"新闻类型：{doc_type}\n"
            f"发布日期：{publish_date}\n"
            f"正文：\n{full_text}"
        )

        parts.append(text)

    return "\n\n".join(parts)


def convert_file(source_name, target_name):

    source = DATA_DIR / source_name
    target = DATA_DIR / target_name

    results = []

    with source.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            sample = json.loads(line)

            docs = sample["docs"]

            output = {
                "issue_list": sample["issue_list"],
                "future_argument": sample["future_argument"]
            }

            record = {
                "instruction": INSTRUCTION,
                "input": build_input(docs),
                "output": json.dumps(
                    output,
                    ensure_ascii=False,
                    separators=(",", ":")
                )
            }

            results.append(record)

    with target.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"{source_name} -> {target_name}")
    print(f"转换样本数：{len(results)}")


if __name__ == "__main__":

    convert_file(
        "train.jsonl",
        "train_sft.json"
    )

    convert_file(
        "val.jsonl",
        "val_sft.json"
    )

    print("\n数据转换完成！")