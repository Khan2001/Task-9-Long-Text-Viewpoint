"""提示词模板：议题抽取 + 未来论点预测。

约定来自训练集统计：
- issue_name 3-16 字名词短语，issue_list[0] 为主议题，其余为子议题，通常 4-5 条
- stance 分布 support 63% / neutral 34% / oppose 2.5%，oppose 仅用于明确批评或分歧
- argument_chain 每议题 1-3 段、单段约 20-80 字，且须为原文逐字片段
- future_argument 与 issue_list 同序等长，单条约 60-110 字
"""

import json

STANCE_RULE = (
    "stance 只能取 support / oppose / neutral：\n"
    "  support=对该议题所描述的方向持肯定、推动、积极评价；\n"
    "  neutral=客观陈述进展、安排、事实，无明显褒贬或仅作中性通报；\n"
    "  oppose=仅当文本明确表达批评、反对、质疑、不满、分歧或划定底线时使用，"
    "措辞委婉但立场性的提醒、关切不算 oppose（此类标 neutral）。"
)

EXTRACT_SYSTEM = (
    "你是一名谈判情报分析师，从给定长文本中挖掘与谈判议题相关的观点三元组"
    "（议题名称、立场倾向、原文证据链）。严格按要求输出 JSON，不输出任何解释文字。\n"
    + STANCE_RULE
)


def extract_prompt(doc_type, publish_date, numbered_text, n_sents, examples=None):
    few = ""
    if examples:
        few = "\n以下是同类文本的标注范例，仅学习其粒度、命名与立场用法，不得照抄内容：\n"
        for ex in examples:
            few += json.dumps(ex, ensure_ascii=False) + "\n"
    return f"""待分析文本（doc_type={doc_type}，publish_date={publish_date}）已由 [1] 到 [{n_sents}] 逐句编号：
<<<TEXT
{numbered_text}
TEXT>>>

任务：识别主议题与相关子议题，输出 JSON。

字段要求：
1) issue_name：3-16 字中文名词短语，概括该议题；第 1 个必须是覆盖全文整体主题的主议题，其余为子议题，共 3-5 个。
2) argument_chain：把上面编号列表中的证据句挑出来，用句子编号数组表示，如 [3, 4, 9]。要求：
   - 每个议题 1-3 条编号，优先只给最关键的 1-2 句；
   - 必须是支撑该议题立场的原句，不得改写、不得编造、不得跨议题重复使用同一句；
   - 主议题的证据应选最能体现整体定调的句子。
3) {STANCE_RULE.replace(chr(10), chr(10) + "   ")}
{few}
输出格式（仅 JSON）：
{{"issue_list": [{{"issue_name": "...", "stance": "support", "evidence_ids": [1, 2]}}, ...]}}"""


def future_prompt(doc_type, full_text, issue_list, examples=None):
    few = ""
    if examples:
        few = "\n范例（左：已抽取议题与立场 右：对应的未来论点），仅学习写法：\n"
        for ex in examples:
            few += json.dumps(ex, ensure_ascii=False) + "\n"
    compact = [
        {"issue_name": it["issue_name"], "stance": it["stance"],
         "evidence": [e[:60] for e in it.get("argument_chain", [])]}
        for it in issue_list
    ]
    return f"""以下是一篇 {doc_type} 类文本：
<<<TEXT
{full_text}
TEXT>>>

已从中抽取的议题链（顺序即输出顺序）：
{json.dumps(compact, ensure_ascii=False, indent=1)}

任务：预判对手/各方在该议题脉络下后续可能表达的观点与论证方向。
要求：
1) future_argument 是字符串数组，长度必须等于议题数量 {len(issue_list)}，且与议题逐项同序对应。
2) 每条 60-110 字，中文，写法为"对该议题的判断或意义 + 后续应/将采取的具体做法（2-3 个抓手）"，语气为政策研判而非建议信。
3) 只能由上文文本的事实与立场推演，不得引入文本中没有的具体事实、数字或专有名词。
4) 同一议题若已标 oppose，预测应延续其分歧与底线表达；标 neutral 则延续客观进展与机制化表述。
{few}
输出格式（仅 JSON）：
{{"future_argument": ["围绕第1个议题的预测", "围绕第2个议题的预测", ...]}}"""


def parse_json_block(raw):
    """从模型输出里稳健地取出第一个 JSON 对象。"""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                for cand in (text[start:i + 1],):
                    try:
                        return json.loads(cand)
                    except json.JSONDecodeError:
                        pass
                return None
    try:
        return json.loads(text[start:] + "}")
    except json.JSONDecodeError:
        return None
