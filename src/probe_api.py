"""密钥连通性探针：依次试几个候选网关，只报状态码与错误类型，不打印密钥。

用法:
    python src/probe_api.py
    python src/probe_api.py --base-url https://你的网关/v1 --model qwen3-32b
    python src/probe_api.py --api-style deepseek --model deepseek-flash
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_api import DEEPSEEK_BASE_URL, load_dotenv  # noqa: E402

CANDIDATES = [
    ("百炼·国内", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3-32b"),
    ("百炼·国际", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "qwen3-32b"),
    ("千问AI平台·兼容模式", "https://maas.qianwenaiapi.com/compatible-mode/v1", "qwen3-32b"),
]


def try_one(name, base_url, model, key, extra_body):
    from openai import OpenAI

    try:
        cli = OpenAI(api_key=key, base_url=base_url, timeout=60)
        r = cli.chat.completions.create(model=model,
                                        messages=[{"role": "user", "content": "只回复两个字：正常"}],
                                        max_tokens=16, extra_body=extra_body)
        return f"OK   {name} | {model} -> {r.choices[0].message.content!r}"
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "status_code", None)
        msg = str(exc).replace(key, "<KEY>")
        return f"FAIL {name} | http={code} | {msg[:180]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-style", choices=["dashscope", "deepseek"], default="dashscope")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    args = ap.parse_args()

    load_dotenv()
    if args.api_style == "deepseek":
        key_var = "DEEPSEEK_API_KEY"
        default_url = DEEPSEEK_BASE_URL
        base_url_var = "DEEPSEEK_BASE_URL"
        default_model = "deepseek-flash"
        extra_body = {"thinking": {"type": "disabled"}}
    else:
        key_var = "DASHSCOPE_API_KEY"
        default_url = CANDIDATES[0][1]
        base_url_var = "DASHSCOPE_BASE_URL"
        default_model = "qwen3-32b"
        extra_body = {"enable_thinking": False}

    key = os.environ.get(key_var, "")
    if not key:
        sys.exit(f".env 里没有 {key_var}")
    print(f"密钥长度 {len(key)}，前缀 {key[:5]}…（完整值不打印）")
    if args.api_style == "deepseek":
        probes = [("DeepSeek", args.base_url or os.environ.get(base_url_var) or default_url,
                   args.model or default_model)]
    else:
        probes = CANDIDATES
        if args.base_url:
            probes = [("自定义网关", args.base_url, args.model or default_model)]
    for name, url, model in probes:
        print(try_one(name, url, model, key, extra_body))


if __name__ == "__main__":
    main()
