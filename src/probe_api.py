"""密钥连通性探针：依次试几个候选网关，只报状态码与错误类型，不打印密钥。

用法:
    python src/probe_api.py
    python src/probe_api.py --base-url https://你的网关/v1 --model qwen3-32b
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_api import load_dotenv  # noqa: E402

CANDIDATES = [
    ("百炼·国内", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3-32b"),
    ("百炼·国际", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "qwen3-32b"),
    ("千问AI平台·兼容模式", "https://maas.qianwenaiapi.com/compatible-mode/v1", "qwen3-32b"),
]


def try_one(name, base_url, model, key):
    from openai import OpenAI

    try:
        cli = OpenAI(api_key=key, base_url=base_url, timeout=60)
        r = cli.chat.completions.create(model=model,
                                        messages=[{"role": "user", "content": "只回复两个字：正常"}],
                                        max_tokens=16, extra_body={"enable_thinking": False})
        return f"OK   {name} | {model} -> {r.choices[0].message.content!r}"
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "status_code", None)
        msg = str(exc).replace(key, "<KEY>")
        return f"FAIL {name} | http={code} | {msg[:180]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    args = ap.parse_args()

    load_dotenv()
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not key:
        sys.exit(".env 里没有 DASHSCOPE_API_KEY")
    print(f"密钥长度 {len(key)}，前缀 {key[:5]}…（完整值不打印）")
    probes = CANDIDATES
    if args.base_url:
        probes = [("自定义网关", args.base_url, args.model or "qwen3-32b")]
    for name, url, model in probes:
        print(try_one(name, url, model, key))


if __name__ == "__main__":
    main()
