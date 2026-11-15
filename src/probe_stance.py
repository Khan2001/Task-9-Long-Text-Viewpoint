#!/usr/bin/env python3
"""立场探针（GPU，短输出，预计 30-40 分钟）：一次跑完三种提问方式，直接量化
"逐议题独立判断"相对"整篇联合判断"的立场准确率差异，并复现塌缩现象。

设计要点（与 predict_test_vllm.py 保持一致，保证可迁移）：
  - 同一基座 /root/.../Qwen3-32B + 同一 adapter outputs/qwen3-32b-qlora，4-bit bitsandbytes；
  - 同一 chat 模板 enable_thinking=False，同一 instruction 取自 train_sft.json[0]；
  - prompt 顺序为 instruction → 正文 → 议题部分，让同一样本的 5 个议题共享前缀，
    配合 vLLM 前缀缓存把 prefill 成本降到约 1/4；
  - 所有请求一次性提交给 llm.generate()，由 continuous batching 吃满并发
    （原脚本按 batch-size 2 循环取模，这是它 300 条跑 5 小时的主因）。

变体：
  I  逐议题 + 该议题论据链（独立判断，短输出）
  N  逐议题 + 无论据（只看原文与议题名，检验论据是否带来增量）
  J  整篇联合：一次给出全部议题，要求按序输出立场数组（复刻联合生成的塌缩）
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

ROOT = Path('/workspace/qwen3-finetune')
BASE = Path('/root/eb-public/huggingface-models/Qwen/Qwen3-32B')
ADAPTER = ROOT / 'outputs/qwen3-32b-qlora'
LABELS = ('support', 'oppose', 'neutral')

HEAD = """你是一名谈判对手观点研判助手。请阅读新闻材料，判断材料中该方对指定议题的立场。
立场取值只能是：
- support：明确支持、肯定、倡导、积极推动；
- oppose：明确反对、批评、质疑、抵制；
- neutral：客观陈述事实、各方表态并陈、无明显倾向。
只输出一个立场词（support/oppose/neutral），不要输出任何其他文字。
"""
JOINT_HEAD = """你是一名谈判对手观点研判助手。请阅读新闻材料，对列出的每个议题分别判断该方的立场。
立场取值只能是 support、oppose、neutral，三者选一。
请按议题顺序输出 JSON 数组，不要输出其他内容，例如：{"stances": ["support", "neutral"]}
"""


def build_doc(docs):
    return '\n\n'.join(
        f'【新闻{i}】\n文档编号：{d.get("doc_id", "")}\n新闻类型：{d.get("doc_type", "")}\n'
        f'发布日期：{d.get("publish_date", "")}\n正文：\n{d.get("full_text", "")}'
        for i, d in enumerate(docs, 1))


def task_i(doc, name, chain):
    ev = '\n'.join(f'- {a}' for a in chain)
    return f"{HEAD}\n【新闻材料】\n{doc}\n\n【议题】{name}\n【该议题的相关论据】\n{ev}\n\n立场："


def task_n(doc, name):
    return f"{HEAD}\n【新闻材料】\n{doc}\n\n【议题】{name}\n\n立场："


def task_j(doc, items):
    lines = '\n'.join(f'{k + 1}. {it["issue_name"]}' for k, it in enumerate(items))
    return f"{JOINT_HEAD}\n【新闻材料】\n{doc}\n\n【议题列表】\n{lines}\n\n输出："


def parse_label(text):
    t = (text or '').strip().lower()
    for lab in LABELS:
        if t.startswith(lab):
            return lab
    m = re.search(r'\b(support|oppose|neutral)\b', t)
    return m.group(1) if m else None


def parse_stances(text, n):
    found = re.findall(r'"(support|oppose|neutral)"', (text or '').lower())
    if len(found) < n:
        found += [None] * (n - len(found))
    return found[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', type=Path, default=ROOT / 'data/val.jsonl')
    ap.add_argument('--train-sft', type=Path, default=ROOT / 'data/train_sft.json')
    ap.add_argument('--adapter', type=Path, default=ADAPTER)
    ap.add_argument('--out', type=Path, default=ROOT / 'probe_stance.jsonl')
    ap.add_argument('--variants', default='I,N,J')
    ap.add_argument('--limit', type=int, default=0, help='仅处理前 N 条样本，用于冒烟测试')
    ap.add_argument('--max-num-seqs', type=int, default=8)
    ap.add_argument('--max-model-len', type=int, default=8192)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.88)
    ap.add_argument('--max-new-tokens', type=int, default=8)
    args = ap.parse_args()
    variants = [v.strip().upper() for v in args.variants.split(',')]

    with args.val.open(encoding='utf-8') as f:
        samples = [json.loads(x) for x in f if x.strip()]
    if args.limit:
        samples = samples[:args.limit]
    instruction = json.load(args.train_sft.open(encoding='utf-8'))[0]['instruction']
    print(f'样本 {len(samples)} 条，变体 {variants}', flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), trust_remote_code=True,
                                              local_files_only=True)

    jobs = []           # (meta, prompt_text)
    for s in samples:
        doc = build_doc(s['docs'])
        items = s['issue_list']
        if 'J' in variants:
            jobs.append(({'kind': 'J', 'sample_id': s['sample_id'], 'n': len(items),
                          'gold': [i['stance'] for i in items]},
                         instruction + '\n' + task_j(doc, items)))
        for idx, it in enumerate(items):
            if 'I' in variants:
                jobs.append(({'kind': 'I', 'sample_id': s['sample_id'], 'idx': idx,
                              'gold': it['stance']},
                             instruction + '\n' + task_i(doc, it['issue_name'], it['argument_chain'])))
            if 'N' in variants:
                jobs.append(({'kind': 'N', 'sample_id': s['sample_id'], 'idx': idx,
                              'gold': it['stance']},
                             instruction + '\n' + task_n(doc, it['issue_name'])))

    built, skipped = [], []
    for meta, text in jobs:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) + args.max_new_tokens >= args.max_model_len:
            skipped.append(meta)
            continue
        built.append((meta, ids))
    print(f'请求 {len(built)} 条，因超长跳过 {len(skipped)} 条', flush=True)
    if args.limit and skipped:
        print('示例:', skipped[:3], flush=True)

    kw = dict(model=str(BASE), tokenizer=str(BASE), quantization='bitsandbytes',
              load_format='bitsandbytes', enable_lora=True, max_lora_rank=16,
              tensor_parallel_size=1, dtype='bfloat16',
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
              trust_remote_code=True)
    try:
        llm = LLM(enable_prefix_caching=True, **kw)
        print('前缀缓存已开启', flush=True)
    except TypeError:
        llm = LLM(**kw)
        print('该 vLLM 版本不支持显式前缀缓存参数，按默认处理', flush=True)

    lora = LoRARequest('probe_lora', 1, str(args.adapter))
    params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens,
                            logprobs=5, stop_token_ids=[tokenizer.eos_token_id])
    start = time.time()
    outputs = llm.generate([{'prompt_token_ids': ids} for _, ids in built],
                           sampling_params=params, lora_request=lora, use_tqdm=False)
    print(f'生成完成，总耗时 {(time.time() - start) / 60:.1f} 分钟', flush=True)

    tmp = args.out.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for (meta, _), out in zip(built, outputs):
            text = out.outputs[0].text if out.outputs else ''
            rec = dict(meta)
            lp = {}
            if out.outputs and out.outputs[0].logprobs:
                for tid, entry in out.outputs[0].logprobs[0].items():
                    lp[tokenizer.decode([tid])] = round(float(entry.logprob), 3)
            rec.update({'raw': text.strip()[:120], 'first_token_logprobs': lp})
            if meta['kind'] == 'J':
                rec['pred'] = parse_stances(text, meta['n'])
            else:
                rec['pred'] = parse_label(text)
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    os.replace(tmp, args.out)
    print(f'写出 {args.out}', flush=True)
    print('把这个文件下载回本地，用 analyze_probe.py 出结论。', flush=True)


if __name__ == '__main__':
    main()
