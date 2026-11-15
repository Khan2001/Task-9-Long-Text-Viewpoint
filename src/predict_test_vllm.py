#!/usr/bin/env python3
"""与原 predict_test.py 保持一致的 vLLM 0.10.2 批量预测脚本。"""
import argparse
import json
import os
import re
import time
from pathlib import Path

ROOT = Path('/workspace/qwen3-finetune')
BASE = Path('/root/eb-public/huggingface-models/Qwen/Qwen3-32B')
ADAPTER = ROOT / 'outputs/qwen3-32b-qlora'
TEST = ROOT / 'data/test.jsonl'
TRAIN_SFT = ROOT / 'data/train_sft.json'
RESULT = ROOT / 'result.jsonl'
OLD_PROGRESS = ROOT / 'test_predictions.progress.jsonl'
PROGRESS = ROOT / 'test_predictions_vllm.progress.jsonl'
VALID_STANCES = {'support', 'oppose', 'neutral'}

def load_jsonl(path):
    records = []
    with path.open('r', encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception as exc:
                    raise ValueError(f'{path}:{number} 不是合法 JSON: {exc}') from exc
    return records

def append_jsonl(path, obj):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')) + '\n')
        f.flush()
        os.fsync(f.fileno())

def build_input(docs):
    # 与 convert_dataset.py 展示出的格式保持一致；若转换脚本有其他处理，需先核对。
    return '\n\n'.join(
        f'【新闻{i}】\n'
        f'文档编号：{doc.get("doc_id", "")}\n'
        f'新闻类型：{doc.get("doc_type", "")}\n'
        f'发布日期：{doc.get("publish_date", "")}\n'
        f'正文：\n{doc.get("full_text", "")}'
        for i, doc in enumerate(docs, 1)
    )

def extract_object(text):
    s = text.strip()
    s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*```$', '', s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start = s.find('{')
        if start == -1:
            raise ValueError('找不到 JSON 对象')
        obj, _ = json.JSONDecoder().raw_decode(s[start:])
        return obj

def validate(sample_id, obj):
    if not isinstance(obj, dict):
        raise ValueError('输出不是对象')
    issues = obj.get('issue_list')
    futures = obj.get('future_argument')
    if not isinstance(issues, list) or not issues:
        raise ValueError('issue_list 必须是非空数组')
    if not isinstance(futures, list) or len(futures) != len(issues):
        raise ValueError('future_argument 数量必须等于 issue_list')
    clean = []
    for i, item in enumerate(issues):
        if not isinstance(item, dict):
            raise ValueError(f'issue_list[{i}] 不是对象')
        name, stance, chain = item.get('issue_name'), item.get('stance'), item.get('argument_chain')
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f'issue_list[{i}].issue_name 无效')
        if stance not in VALID_STANCES:
            raise ValueError(f'issue_list[{i}].stance 无效: {stance}')
        if not isinstance(chain, list) or not chain or not all(isinstance(x, str) and x.strip() for x in chain):
            raise ValueError(f'issue_list[{i}].argument_chain 无效')
        clean.append({'issue_name': name.strip(), 'stance': stance, 'argument_chain': [x.strip() for x in chain]})
    if not all(isinstance(x, str) and x.strip() for x in futures):
        raise ValueError('future_argument 必须全部为非空字符串')
    return {'sample_id': sample_id, 'issue_list': clean, 'future_argument': [x.strip() for x in futures]}


def load_completed(samples, old_progress, progress):
    allowed = {s['sample_id'] for s in samples}
    done = {}
    for path in (old_progress, progress):
        if not path.exists():
            continue
        n = 0
        for row in load_jsonl(path):
            sid = row.get('sample_id') if isinstance(row, dict) else None
            if sid not in allowed:
                raise ValueError(f'{path} 存在未知 sample_id: {sid}')
            done[sid] = validate(sid, row)
            n += 1
        print(f'读取进度 {path.name}: {n} 行；去重后累计 {len(done)} 条', flush=True)
    return done


def write_final(samples, done, output):
    ids = [s['sample_id'] for s in samples]
    missing = [sid for sid in ids if sid not in done]
    if missing:
        print(f'尚缺 {len(missing)} 条，不生成不完整的正式结果。示例: {missing[:5]}', flush=True)
        return False
    tmp = output.with_name(output.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        for sid in ids:
            f.write(json.dumps(validate(sid, done[sid]), ensure_ascii=False, separators=(',', ':')) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, output)
    print(f'全部 {len(ids)} 条通过校验，已生成 {output}', flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter', type=Path, default=ADAPTER)
    p.add_argument('--test', type=Path, default=TEST)
    p.add_argument('--train', type=Path, default=TRAIN_SFT)
    p.add_argument('--output', type=Path, default=RESULT)
    p.add_argument('--progress', type=Path, default=PROGRESS)
    p.add_argument('--old-progress', type=Path, default=OLD_PROGRESS)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--max-samples', type=int, default=0, help='与旧脚本一致：仅处理测试集前 N 条；0=全部')
    p.add_argument('--max-input-tokens', type=int, default=6144)
    p.add_argument('--max-new-tokens', type=int, default=3072)
    p.add_argument('--max-model-len', type=int, default=8192)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    p.add_argument('--retries', type=int, default=2)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    if args.batch_size < 1 or args.max_samples < 0 or args.retries < 0:
        p.error('batch-size 必须>=1，max-samples 和 retries 必须>=0')
    if args.max_input_tokens < 1 or args.max_new_tokens < 1 or args.max_model_len < 2:
        p.error('token 限制必须为正整数')
    if not 0 < args.gpu_memory_utilization <= 1:
        p.error('gpu-memory-utilization 必须在 (0,1]')

    samples = load_jsonl(args.test)
    if not samples:
        raise ValueError('测试集为空')
    ids = [s['sample_id'] for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError('测试集 sample_id 存在重复')
    if any(not isinstance(s.get('docs'), list) for s in samples):
        raise ValueError('测试集有样本缺少 docs 数组')
    done = load_completed(samples, args.old_progress, args.progress)
    if args.check_only:
        write_final(samples, done, args.output)
        return
    targets = samples[:args.max_samples] if args.max_samples else samples
    pending = [s for s in targets if s['sample_id'] not in done]
    print(f'测试集 {len(samples)} 条，本次目标 {len(targets)} 条，已完成 {len(targets)-len(pending)} 条，待推理 {len(pending)} 条', flush=True)
    if not pending:
        write_final(samples, done, args.output)
        return

    if not (args.adapter / 'adapter_config.json').is_file():
        raise FileNotFoundError(f'缺少 LoRA 配置: {args.adapter}')
    if not args.train.is_file():
        raise FileNotFoundError(f'缺少训练指令: {args.train}')
    with args.train.open('r', encoding='utf-8') as f:
        instruction = json.load(f)[0]['instruction']
    print('使用 train_sft.json 第一条 instruction，与旧脚本一致。', flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), trust_remote_code=True, local_files_only=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError('Tokenizer 缺少 eos_token_id')
    prompts = {}
    failures = args.progress.with_name('test_predictions_vllm.failures.jsonl')
    runnable = []
    for sample in pending:
        sid = sample['sample_id']
        content = build_input(sample['docs'])
        messages = [{'role': 'user', 'content': instruction + '\n' + content}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        n = len(token_ids)
        if n > args.max_input_tokens or n >= args.max_model_len:
            error = f'输入 {n} tokens，超过输入限制 {args.max_input_tokens} 或上下文 {args.max_model_len}；未截断'
            append_jsonl(failures, {'sample_id': sid, 'error': error})
            print(f'[跳过] {sid}: {error}', flush=True)
            continue
        prompts[sid] = (prompt, n)
        runnable.append(sample)
    if not runnable:
        write_final(samples, done, args.output)
        return
    print(f'输入 token 长度: {min(n for _,n in prompts.values())}~{max(n for _,n in prompts.values())}', flush=True)

    # 仅在需要推理且输入校验通过后加载大模型。
    print('开始加载 vLLM 4-bit Qwen3-32B + LoRA ...', flush=True)
    llm = LLM(
        model=str(BASE), tokenizer=str(BASE),
        quantization='bitsandbytes', load_format='bitsandbytes',
        enable_lora=True, max_lora_rank=16,
        tensor_parallel_size=1, dtype='bfloat16',
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, max_num_seqs=args.batch_size,
        trust_remote_code=True,
    )
    lora = LoRARequest('competition_lora', 1, str(args.adapter))
    params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, stop_token_ids=[eos_id])

    def generate(batch):
        # 用 token_ids 避免 tokenizer 二次处理，并保证与旧脚本的输入一致。
        requests = [{'prompt_token_ids': tokenizer.encode(prompts[s['sample_id']][0], add_special_tokens=False)} for s in batch]
        return llm.generate(requests, sampling_params=params, lora_request=lora, use_tqdm=False)

    start = time.time()
    for offset in range(0, len(runnable), args.batch_size):
        batch = runnable[offset:offset + args.batch_size]
        try:
            outputs = generate(batch)
            if len(outputs) != len(batch):
                raise RuntimeError('vLLM 返回数量与请求数量不一致')
        except Exception as exc:
            print(f'[批次失败] {exc}；尝试逐条重试', flush=True)
            outputs = [None] * len(batch)
        for sample, output in zip(batch, outputs):
            sid = sample['sample_id']
            success = False
            for attempt in range(args.retries + 1):
                text = ''
                try:
                    if attempt or output is None:
                        output = generate([sample])[0]
                    if not output.outputs:
                        raise ValueError('vLLM 未返回文本')
                    generated = output.outputs[0]
                    text = generated.text
                    if generated.finish_reason == 'length':
                        raise ValueError('输出达到长度上限或上下文限制，可能被截断')
                    result = validate(sid, extract_object(text))
                    append_jsonl(args.progress, result)
                    done[sid] = result
                    print(f'[成功] {sid} | 输出 {len(generated.token_ids)} tokens | 累计 {len(done)}/{len(samples)}', flush=True)
                    success = True
                    break
                except Exception as exc:
                    append_jsonl(failures, {'sample_id': sid, 'attempt': attempt+1, 'error': str(exc), 'raw_output': text[:16000]})
                    print(f'[失败] {sid} 第 {attempt+1} 次: {exc}', flush=True)
            if not success:
                print(f'[待重跑] {sid}', flush=True)
        print(f'本轮已处理 {min(offset+len(batch),len(runnable))}/{len(runnable)}；耗时 {(time.time()-start)/60:.1f} 分钟', flush=True)

    if not write_final(samples, done, args.output):
        print(f'未完成条目可以原命令断点续跑。失败日志: {failures}', flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
