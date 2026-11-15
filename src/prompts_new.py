"""提示词模板（实验版 · LangGPT 结构化）：议题抽取 + 未来论点预测。

改写要点：
1) 采用 LangGPT 式分节结构（# Role / ## Profile / ## Rules / ## Workflow /
   ## Few-shot / ## OutputFormat），让模型按角色—约束—流程—格式的顺序理解任务。
2) 内容沿用基于 val 失分诊断得到的改进规则：
   - 主议题允许/鼓励上位总括表述（对齐 gold 命名风格），仅子议题禁止上位大词；
   - 新增“禁止过度拆分”“宁少勿滥”；
   - 收紧 oppose（明确负面立场才算）。
   诊断依据：val 上 F1_ext=0.657 为瓶颈；未匹配议题中 25.4% 因相似度不足
   （平均 sim_议题名仅 0.514），8.5% 因立场门（oppose 多判 63 vs gold 42）。

对外接口与 prompts.py 完全一致：extract_prompt / future_prompt / parse_json_block，
可在 infer_api.py 中通过切换导入模块做 A/B。
"""

import json

# ---------------------------------------------------------------------------
# 可复用规则块（供 system 与 user 提示词共同引用）
# ---------------------------------------------------------------------------

STANCE_RULE = (
    "stance 只能取 support / oppose / neutral，判定时优先按“有无明确负面立场”筛查：\n"
    "- oppose：仅当文本对该议题出现明确、直接的批评、反对、拒绝、强烈质疑或划定对抗性底线时使用；"
    "这是少数情况（全文占比通常很低），拿不准时一律不选 oppose。"
    "以下都不算 oppose，应标 neutral：措辞委婉的提醒/关切、客观转述他方分歧、"
    "“仍有分歧/未达成共识”这类事实性通报、以及仅陈述谈判难点而无褒贬的表达。\n"
    "- support：对该议题所描述的方向持肯定、推动、积极评价。\n"
    "- neutral：客观陈述进展、安排、事实，无明显褒贬，或仅作中性通报。"
)

NAMING_RULES = (
    "议题命名要求（评分重点，务必严格遵守）：\n"
    "- 议题名必须是简洁的中文名词短语，不能写成句子；优先复用原文的中心名词并做适度概括。\n"
    "- 【主议题】第 1 个是覆盖全文整体主题的主议题（10-14 字）。它应当是全文的总括性主题，"
    "允许并鼓励使用上位/主题式表述（如“……深化”“……常态化长效化”“……合作与安全”"
    "“科技自立自强”“高端装备自主化进展”），可直接借鉴原文标题或反复出现的核心概念；"
    "切勿把某个吸引眼球的局部细节当作主议题。\n"
    "- 【子议题】其余为具体子议题（8-11 字），粒度取“事项+动作/领域”的中间层，"
    "正例：“eVTOL技术引进”“风电项目投资”“中文教育推广”“革命文物数字化保护”“督查结果奖惩机制”；"
    "子议题避免过泛的上位大词（如“科技创新合作”“文化交流活动”），"
    "也不得堆砌细节：不出现具体数字、金额、年份、日期、公司名、人名、地名，"
    "不把一整件事压成标题（如“峰飞航空eVTOL在阿拉木图首飞”应写成“eVTOL技术引进”）。\n"
    "- 【禁止过度拆分】若多个具体事项同属一个主题，必须合并为一个议题，"
    "不要拆成 2-3 条并列的细节议题；只挑选全文最核心的议题，宁少勿滥。\n"
    "- 议题总数通常 4-5 个（1 主 + 3-4 子）；只有当全文确有多条相互独立的重要脉络时才给 5 个，"
    "主题较集中时给 4 个即可，不要为凑数而细分。"
)

EVIDENCE_RULE = (
    "证据（argument_chain / evidence_ids）要求：\n"
    "- 每个议题给 1-3 条句子编号，优先只给最关键的 1-2 句；\n"
    "- 必须是支撑该议题立场的原句，不得改写、不得编造、不得跨议题重复使用同一句；\n"
    "- 主议题的证据应选最能体现全文整体定调的句子。"
)

# ---------------------------------------------------------------------------
# System 提示词（LangGPT 角色定义）
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = f"""# Role: 观点分析师

## Profile
- Language: 中文
- Description: 从给定长文本中挖掘与谈判议题相关的观点三元组（议题名称、立场倾向、原文证据链）。

## Rules
1. 严格按要求输出 JSON，不输出任何解释性文字、前后缀或 markdown 代码块以外的内容。
2. 证据只能引用输入中带编号的原句编号，绝不改写或编造。
3. {STANCE_RULE.replace(chr(10), chr(10) + "   ")}

## Workflow
1. 通读全文 → 2. 定主议题 → 3. 归并子议题 → 4. 判立场 → 5. 选证据句 → 6. 按格式输出。
"""

FUTURE_SYSTEM = """# Role: 谈判对手研判专家

## Profile
- Language: 中文
- Description: 基于给定文本与已抽取的议题链，预判各方后续可能表达的观点与论证方向。

## Rules
1. 只输出 JSON，future_argument 必须与议题数量相同且同序。
2. 只能由上文事实与立场推演，不得引入文本中没有的具体事实、数字或专有名词。
"""


def extract_prompt(doc_type, publish_date, numbered_text, n_sents, examples=None):
    """构造 LangGPT 结构的议题抽取用户提示词。"""
    few_section = ""
    if examples:
        body = "\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples)
        few_section = f"""
## Few-shot
下面是若干同类文本的人工标注范例。你只能学习它们的议题命名粒度、立场用法和证据选取方式，
**绝对不许照抄范例内容**：
{body}
"""
    return f"""# Role: 谈判情报分析师

## Profile
- Language: 中文
- Description: 从下方已逐句编号的长文本中，挖掘主议题与相关子议题的观点三元组。

## Input
- doc_type: {doc_type}
- publish_date: {publish_date}
- 句子总数: {n_sents}（编号 [1] 到 [{n_sents}]）
- 待分析文本:
<<<TEXT
{numbered_text}
TEXT>>>

## Rules
### R1 命名规则
{NAMING_RULES}

### R2 立场规则
{STANCE_RULE}

### R3 证据规则
{EVIDENCE_RULE}

## Workflow
1. 通读全文，识别覆盖整体主题的主议题（放在 issue_list 第 1 位）。
2. 找出 3-4 个相互独立的核心子议题；同属一个主题的细节合并为一条，不过度拆分。
3. 为每个议题按 R2 判定 stance（拿不准不选 oppose）。
4. 按 R3 从编号句中挑选 1-2 句最能支撑该议题立场的原句，填入 evidence_ids。
5. 自检：议题名是否为名词短语、粒度是否合规、总数是否 4-5、证据是否为原文编号。
6. 按 OutputFormat 输出，仅 JSON。
{few_section}
## OutputFormat
仅输出如下 JSON，不要任何解释文字：
{{"issue_list": [{{"issue_name": "...", "stance": "support", "evidence_ids": [1, 2]}}, ...]}}"""


def future_prompt(doc_type, full_text, issue_list, examples=None):
    """构造 LangGPT 结构的未来论点预测用户提示词。"""
    few_section = ""
    if examples:
        body = "\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples)
        few_section = f"""
## Few-shot
范例（左：已抽取议题与立场 右：对应的未来论点），仅学习写法，不得照抄：
{body}
"""
    compact = [
        {"issue_name": it["issue_name"], "stance": it["stance"],
         "evidence": [e[:60] for e in it.get("argument_chain", [])]}
        for it in issue_list
    ]
    return f"""# Role: 谈判对手研判专家

## Profile
- Language: 中文
- Description: 基于给定文本与已抽取的议题链，预判各方在该脉络下后续可能表达的观点与论证方向。

## Input
- doc_type: {doc_type}
- 原文:
<<<TEXT
{full_text}
TEXT>>>
- 已抽取的议题链（顺序即输出顺序）:
{json.dumps(compact, ensure_ascii=False, indent=1)}

## Rules
1. future_argument 是字符串数组，长度必须等于议题数量 {len(issue_list)}，且与议题逐项同序对应。
2. 每条 60-110 字，中文，写法为“对该议题的判断或意义 + 后续应/将采取的具体做法（2-3 个抓手）”，
   语气为政策研判而非建议信。
3. 只能由上文文本的事实与立场推演，不得引入文本中没有的具体事实、数字或专有名词。
4. 同一议题若已标 oppose，预测应延续其分歧与底线表达；标 neutral 则延续客观进展与机制化表述。

## Workflow
1. 逐个议题阅读其立场与证据，把握当前定调。
2. 顺着该定调推演后续最可能表达的观点与做法（2-3 个抓手）。
3. 控制每条 60-110 字，保持政策研判语气。
4. 自检：条数是否等于 {len(issue_list)}、是否同序、有无引入外部事实。
5. 按 OutputFormat 输出，仅 JSON。
{few_section}
## OutputFormat
仅输出如下 JSON，不要任何解释文字：
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
