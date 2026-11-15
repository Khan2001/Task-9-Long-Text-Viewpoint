"""从剪贴板安全写入 .env，避免手抄错字符。

用法:
    python src/set_key.py                 # 读剪贴板并写 DASHSCOPE_API_KEY
    python src/set_key.py --var DEEPSEEK_API_KEY
    python src/set_key.py --value sk-xxx  # 显式传入（不推荐，会进 shell 历史）
只打印长度与首尾几位，不回显完整密钥。
"""

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = os.path.join(ROOT, ".env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", help="直接传入密钥；缺省则读 macOS 剪贴板")
    ap.add_argument("--var", default="DASHSCOPE_API_KEY")
    ap.add_argument("--base-url", default="", help="同时写入自定义网关地址")
    args = ap.parse_args()

    if args.value:
        raw = args.value
    else:
        raw = ""
        if not sys.stdin.isatty():
            try:
                raw = sys.stdin.read()
            except Exception:  # noqa: BLE001
                raw = ""
        if not raw.strip():
            raw = subprocess.run(["pbpaste"], capture_output=True, text=True).stdout or ""
    key = re.sub(r"^.*?=", "", raw.strip(), count=1) if raw.strip().startswith(args.var) else raw.strip()
    key = re.sub(r"^['\"]|['\"]$", "", key)
    key = key.strip().strip("'\"").replace(" ", "")
    if not key:
        sys.exit("剪贴板为空：请到平台 API Key 页面点“复制”按钮后再运行")
    if len(key) < 20 or not key.startswith("sk-"):
        sys.exit(f"剪贴板内容不像 API Key（长度 {len(key)}，开头 {key[:6]!r}）。"
                 "请确认复制的是密钥本身，而不是命令或说明文字。")

    updates = {args.var: key}
    if args.base_url:
        base_var = args.var[:-8] + "_BASE_URL" if args.var.endswith("_API_KEY") else args.var + "_BASE_URL"
        updates[base_var] = args.base_url

    existing = []
    if os.path.exists(ENV):
        with open(ENV, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    name = stripped.split("=", 1)[0].strip()
                    if name in updates:
                        continue
                existing.append(line.rstrip("\n"))

    lines = existing + [f"{name}={value}" for name, value in updates.items()]
    with open(ENV, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(ENV, 0o600)
    print(f"已写入 .env：长度 {len(key)}，头 {key[:6]}…，尾 …{key[-3:]}")
    if args.var == "DEEPSEEK_API_KEY":
        print("自查: python src/probe_api.py --api-style deepseek --model deepseek-flash")
    else:
        print("自查: python src/probe_api.py")


if __name__ == "__main__":
    main()
