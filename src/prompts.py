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


NAMING_RULES = (
    "议题命名要求（评分重点，务必严格遵守）：\n"
    "  - 第 1 个是覆盖全文整体主题的主议题（10-13 字），其余为具体子议题（8-11 字）；"
    "必须是简洁的中文名词短语，不能写成句子；\n"
    "  - 粒度取“事项+动作/领域”的中间层，正例：“eVTOL技术引进”“风电项目投资”“中文教育推广”"
    "“革命文物数字化保护”“督查结果奖惩机制”；\n"
    "  - 禁止上位大词（太泛），例如“科技创新合作”“能源项目合作”“文化交流活动”“红色资源利用与实践创新”；\n"
    "  - 禁止堆砌细节（太细）：议题名里不得出现具体数字、金额、年份、日期、公司名、人名、地名，"
    "也不得把一整件事压成标题，例如不能写“峰飞航空eVTOL在阿拉木图首飞”“卡拉干达州500MW风储项目开工”"
    "“全国革命文物数字化保护工程启动”，应分别写成“eVTOL技术引进”“风电项目投资”“革命文物数字化保护”；\n"
    "  - 优先复用原文关键名词，但要做适度概括，最终长度必须落在上面的区间内；\n"
    "  - 议题总数通常 5 个（1 主 + 4 子），原文内容确实不足时可 4 个。"
)


def extract_prompt(doc_type, publish_date, numbered_text, n_sents, examples=None):
    few = ""
    if examples:
        few = ("\n下面是若干同类文本的人工标注范例。你只能学习它们的议题命名粒度、"
               "立场用法和证据选取方式，绝对不许照抄范例内容：\n")
        for ex in examples:
            few += json.dumps(ex, ensure_ascii=False) + "\n"
    return f"""待分析文本（doc_type={doc_type}，publish_date={publish_date}）已由 [1] 到 [{n_sents}] 逐句编号：
<<<TEXT
{numbered_text}
TEXT>>>

任务：识别主议题与相关子议题，输出 JSON。

{NAMING_RULES}

字段要求：
1) issue_name：按上面的命名与粒度要求，给出中文名词短语。
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
