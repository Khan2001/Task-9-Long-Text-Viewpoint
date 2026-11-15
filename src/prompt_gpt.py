# -*- coding: utf-8 -*-
"""议题抽取与未来论点预测 Prompt 模板。

兼容原有接口：
    extract_prompt(doc_type, publish_date, numbered_text, n_sents,
                   examples=None, issue_names=None)
    future_prompt(doc_type, full_text, issue_list, examples=None)
    parse_json_block(raw)

推荐流程：
1. 将原文逐句编号后，调用 extract_prompt。
2. 解析模型返回的 JSON，并用 materialize_extract_result 将 evidence_ids
   转换为原文中的 argument_chain，确保论据是可追溯的原文片段。
3. 将转换后的 issue_list 传给 future_prompt。
4. 解析 future_argument，并用 validate_future_result 检查输出长度与类型。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


VALID_STANCES = {"support", "oppose", "neutral"}

STANCE_RULE = """立场只能取以下三个英文值，必须严格拼写：
- support：文本对该议题所代表的方向、做法、主张或发展趋势表达肯定、支持、推动、认可，或明确呈现积极成效。
- oppose：文本明确反对、批评、质疑、否定，或清晰呈现各方分歧、抵制与不可退让的底线。只有出现实质性反对证据时才使用。
- neutral：文本仅客观介绍事实、进展、安排、表态或背景，无法从证据中确认明确支持/反对；礼貌性提醒、一般关切、尚未表明态度的讨论也归 neutral。
判断对象是“文本针对该议题呈现的立场”，不是判断议题本身好坏，也不能仅凭某个词的情绪色彩定类。
如果同一议题同时出现正反信息，以与该议题直接相关、表达最明确的总体立场为准；无法确定时选 neutral。
类别比例只可作为证据不足时的弱先验，绝不能为了凑比例而改变原文证据。"""

NAMING_RULES = """议题识别与命名：
- 第 1 个是覆盖全文整体主题的【主议题】（10–14 字）。它应是全文的总括性主题，允许并鼓励使用上位/主题式表述
  （如"……深化""……常态化长效化""……合作与安全""科技自立自强""高端装备自主化进展"），
  可直接借鉴原文标题或反复出现的核心概念；切勿把某个吸引眼球的局部细节当作主议题。
- 其余为【子议题】（8–11 字），粒度取"事项+动作/领域"的中间层，
  正例："eVTOL技术引进""风电项目投资""中文教育推广""革命文物数字化保护""督查结果奖惩机制"；
  子议题避免过泛的上位大词（如"科技创新合作""文化交流活动"），也不得堆砌细节：
  议题名里不出现具体数字、金额、年份、日期、公司名、人名、地名，
  不把一整件事压成标题（如"峰飞航空eVTOL在阿拉木图首飞"应写成"eVTOL技术引进"）。
- 议题名必须是简洁的中文名词短语，不能写成句子；优先复用原文的关键名词并做适度概括。
- 【禁止过度拆分】若多个具体事项同属一个主题，必须合并为一个议题，不要拆成 2–3 条并列的细节议题；
  只挑全文最核心的议题，宁少勿滥。
- 议题总数通常 4–5 个（1 主 + 3–4 子）；只有全文确有多条相互独立的重要脉络时才给 5 个，
  主题较集中时给 4 个即可，不得为了凑数制造弱相关议题。
- 议题顺序：第 1 项为主议题，其余按重要性排列。"""

EXTRACT_SYSTEM = """你是一名谈判情报分析师。你要从给定长文本中识别议题、判定文本对各议题呈现的立场，
并定位支持判断的原文证据。必须遵守指定的字段和输出格式：证据只能引用输入中带编号的原句编号，
不得编造原文没有的信息，也不输出解释、分析过程或 Markdown 代码围栏。"""


def _json_dumps(value: Any, indent: Optional[int] = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent)


def _format_examples(
    examples: Optional[Sequence[Dict[str, Any]]], task: str
) -> str:
    """序列化少量调用方提供的范例，不假定范例拥有特定字段结构。"""
    if not examples:
        return ""
    if task == "extract":
        heading = "\n参考范例（仅学习议题粒度、证据选择和立场判定方法；不得复制范例内容）：\n"
    else:
        heading = "\n参考范例（仅学习未来论点的具体程度和表达方式；不得复制范例内容）：\n"
    blocks = []
    for example in list(examples)[:3]:
        serialized = _json_dumps(example)
        if len(serialized) > 9000:
            serialized = serialized[:9000] + "……[范例截断]"
        blocks.append(serialized)
    return heading + "\n".join(blocks) + "\n"


def extract_prompt(
    doc_type: str,
    publish_date: str,
    numbered_text: str,
    n_sents: int,
    examples: Optional[Sequence[Dict[str, Any]]] = None,
    issue_names: Optional[Sequence[str]] = None,
) -> str:
    """构造议题抽取提示词。

    若传入 issue_names（当数据已提供议题名时推荐这样做），提示词会要求模型
    原样保留议题名和顺序，只预测立场与证据。若不传入，则由模型从文本中发现议题。
    """
    examples_text = _format_examples(examples, "extract")

    if issue_names is not None:
        names = [str(name) for name in issue_names]
        issue_instruction = (
            "本样本已经给出待判断的议题名称。你必须逐项使用下列议题名称，字符级原样保留，"
            "严格保持数量和顺序；不得改名、合并、拆分、补充或删除议题：\n"
            + _json_dumps(names, indent=2)
            + "\n只为这些议题判断 stance 并选择证据。即使某个议题证据较弱，也不得自行替换议题名；"
            "仍必须为它选出至少 1 个最直接的句子编号，确实找不到直接证据时选最接近的一句并谨慎判为 neutral。"
        )
        output_count_rule = f"issue_list 必须恰好有 {len(names)} 项，且顺序与给定议题完全一致。"
    else:
        issue_instruction = NAMING_RULES
        output_count_rule = "通常输出 4–5 项；若文本确实只支持更少议题，不得为凑数而编造。"

    return f'''{EXTRACT_SYSTEM}

文本元信息：doc_type={doc_type}；publish_date={publish_date}
文本已按句子编号，共 {n_sents} 句。编号后的句子是唯一合法的证据来源。

<NUMBERED_TEXT>
{numbered_text}
</NUMBERED_TEXT>

任务：识别议题并为每个议题标注文本立场，选择能够直接支撑议题与立场的原文句子编号。

{issue_instruction}

证据选择规则：
1. 每个议题必须选择 1–3 个最相关的句子编号，优先选择直接表达该议题主张、态度、行动或结果的句子，不要为了数量而加入弱相关背景句。
2. 只能输出上文中真实存在的整数编号，范围为 1–{n_sents}。不得改写句子、推测编号或把一个句子拆成多个编号。
3. 每个议题至少要给出 1 个编号；即使证据较弱，也必须选出最接近的那一句，不得返回空的 evidence_ids。
4. 主议题优先选择最能体现全文核心方向的证据；子议题选择其自身最直接的证据。
5. 不得跨议题重复使用同一个句子编号：编号一旦被前面的议题用过，系统会自动从后续议题中剔除，导致该议题证据丢失。
6. 证据应足以支持所选 stance。仅仅提到某议题、不表达态度的句子，通常不足以独立证明 support 或 oppose。
7. 不要把新闻中“某人表示/会议指出”等转述形式自动视为 neutral 或 support；要看被转述的具体内容及其与议题的关系。
8. {output_count_rule}

{STANCE_RULE}

{examples_text}

仅输出合法 JSON 对象，不要附加 Markdown 或解释。结构如下：
{{
  "issue_list": [
    {{
      "issue_name": "议题名称",
      "stance": "support",
      "evidence_ids": [1, 2]
    }}
  ]
}}'''


def future_prompt(
    doc_type: str,
    full_text: str,
    issue_list: Sequence[Dict[str, Any]],
    examples: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """构造未来论点预测提示词；预测条数和顺序严格跟随 issue_list。"""
    examples_text = _format_examples(examples, "future")
    compact: List[Dict[str, Any]] = []

    for index, item in enumerate(issue_list, start=1):
        entry: Dict[str, Any] = {
            "index": index,
            "issue_name": item.get("issue_name", ""),
            "stance": item.get("stance", "neutral"),
        }
        # 与 prompts_new 基线一致：证据只给前 60 字，避免预测阶段提示词膨胀（否则 A/B 混入输入长度差异）
        if item.get("argument_chain"):
            entry["evidence"] = [str(e)[:60] for e in item["argument_chain"]]
        elif item.get("evidence"):
            entry["evidence"] = [str(e)[:60] for e in item["evidence"]]
        elif item.get("evidence_ids"):
            entry["evidence_ids"] = item["evidence_ids"]
        compact.append(entry)

    return f'''你是一名严谨的趋势研判分析员。请依据给定文本与已识别议题，预测各议题后续可能出现的观点、论证重点或行动方向。预测不是随意提出建议，也不是续写新闻；必须建立在原文已经呈现的事实、立场、计划、问题或发展趋势之上。

文本类型：{doc_type}

<FULL_TEXT>
{full_text}
</FULL_TEXT>

已识别议题（顺序固定）：
{_json_dumps(compact, indent=2)}

生成规则：
1. 输出 future_argument 字符串数组，长度必须严格等于 {len(issue_list)}，与以上议题逐项同序对应；不得增项、漏项、换序，也不要重复输出议题名称作为独立字段。
2. 每条建议控制在 60–110 个汉字左右。优先保证与议题、原文的关联和信息密度，不要为满足字数添加空话。
3. 每条写成“基于当前事实/立场的后续判断 + 可能进一步出现的论证或具体进展”。可涉及制度落实、资源配置、实施范围、效果评估、争议焦点或后续协商事项，但只有在原文能支持这种推演时才写。
4. 预测必须具备议题专属性：说清楚“这一议题接下来可能讨论什么、推进什么或争论什么”。避免“进一步加强合作、持续深化交流、促进高质量发展”等放到任何议题都成立的模板句。
5. 只能使用原文提供的事实和合理的低风险推演。不得捏造新的数字、时间、机构、人名、地名、项目名、已经发生的事件或确定性承诺。对不确定的未来使用“可能、预计、或将、重点可能转向”等措辞，不把推测写成已发生的事实。
6. 必须与该议题的 stance 一致：
   - support：沿着文本已表达的支持方向，预测后续落实、扩展、检验成效或论证其价值的可能路径；
   - oppose：延续文本明确呈现的反对、质疑、分歧或底线，预测后续可能关注的争议、条件、风险或妥协空间；不得把反对擅自写成支持；
   - neutral：以客观进展、待解决事项、后续安排或需要进一步确认的信息为主，不擅自虚构支持或反对立场。
7. 未来论点是“可能出现的观点/论证方向”，不是直接向读者发出的政策建议。因此少用“应该、必须、建议你”等指令式措辞，优先描述各方可能提出的论据、关注点和下一步变化。
8. 不得将同一套表述简单复制到所有议题。每条都应包含能从该议题及原文中识别出来的具体名词、行动或问题。
9. 输入中的议题名和立场是上游结果；不要在本步骤重新分类、改名或重新排序。

{examples_text}

仅输出合法 JSON 对象，不要附加 Markdown 或解释。结构如下：
{{
  "future_argument": [
    "第 1 个议题对应的未来论点",
    "第 2 个议题对应的未来论点"
  ]
}}'''


def parse_json_block(raw: str) -> Optional[Dict[str, Any]]:
    """从纯文本或 Markdown 围栏模型输出中解析第一个 JSON 对象。"""
    if not raw or not isinstance(raw, str):
        return None
    start = raw.find("{")
    if start < 0:
        return None
    try:
        value, _end = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_numbered_sentences(numbered_text: str) -> Dict[int, str]:
    """将 '[1] 句子\n[2] 句子' 转成 {1: '句子', 2: '句子'}。"""
    result: Dict[int, str] = {}
    current_id: Optional[int] = None
    current_parts: List[str] = []
    line_pattern = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")

    for line in numbered_text.splitlines():
        match = line_pattern.match(line)
        if match:
            if current_id is not None:
                result[current_id] = "\n".join(current_parts).strip()
            current_id = int(match.group(1))
            current_parts = [match.group(2)]
        elif current_id is not None:
            current_parts.append(line)

    if current_id is not None:
        result[current_id] = "\n".join(current_parts).strip()
    return result


def materialize_extract_result(
    result: Dict[str, Any],
    numbered_text: str,
    issue_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """把 evidence_ids 映射回原文句子，产出带 argument_chain 的 issue_list。

    若提供 issue_names，则强制使用这些名称和顺序。无效编号、重复编号会被忽略。
    """
    sentences = parse_numbered_sentences(numbered_text)
    raw_items = result.get("issue_list", [])
    if not isinstance(raw_items, list):
        raw_items = []

    if issue_names is not None:
        items: List[Tuple[str, Dict[str, Any]]] = []
        names = [str(name) for name in issue_names]
        for i, name in enumerate(names):
            source = raw_items[i] if i < len(raw_items) and isinstance(raw_items[i], dict) else {}
            items.append((name, source))
    else:
        items = [
            (str(item.get("issue_name", "")).strip(), item)
            for item in raw_items
            if isinstance(item, dict) and str(item.get("issue_name", "")).strip()
        ]

    materialized: List[Dict[str, Any]] = []
    source_text = "\n".join(sentences.values())
    for name, item in items:
        stance = item.get("stance", "neutral")
        if stance not in VALID_STANCES:
            stance = "neutral"

        ids = item.get("evidence_ids", [])
        if not isinstance(ids, list):
            ids = []
        argument_chain: List[str] = []
        seen = set()
        for value in ids:
            try:
                sentence_id = int(value)
            except (TypeError, ValueError):
                continue
            snippet = sentences.get(sentence_id)
            if snippet and snippet not in seen:
                argument_chain.append(snippet)
                seen.add(snippet)
            if len(argument_chain) >= 3:
                break

        # 兼容模型直接返回 argument_chain 的情况，但只保留能在源文中逐字找到的片段。
        if not argument_chain and isinstance(item.get("argument_chain"), list):
            for snippet in item["argument_chain"]:
                if isinstance(snippet, str) and snippet and snippet in source_text:
                    argument_chain.append(snippet)
                if len(argument_chain) >= 3:
                    break

        materialized.append({
            "issue_name": name,
            "stance": stance,
            "argument_chain": argument_chain,
        })

    return {"issue_list": materialized}


def validate_future_result(result: Dict[str, Any], issue_count: int) -> Dict[str, Any]:
    """校正未来论点输出的外部结构；不对语义内容打分或擅自改写。"""
    values = result.get("future_argument", [])
    if not isinstance(values, list):
        values = []
    cleaned = [value if isinstance(value, str) else "" for value in values[:issue_count]]
    if len(cleaned) < issue_count:
        cleaned.extend([""] * (issue_count - len(cleaned)))
    return {"future_argument": cleaned}


__all__ = [
    "STANCE_RULE",
    "NAMING_RULES",
    "EXTRACT_SYSTEM",
    "extract_prompt",
    "future_prompt",
    "parse_json_block",
    "parse_numbered_sentences",
    "materialize_extract_result",
    "validate_future_result",
]
