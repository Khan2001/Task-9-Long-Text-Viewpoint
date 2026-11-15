"""查提交件里的"疑似重复议题"：同一篇内议题名很接近、或证据高度重叠的议题对。

gold 的议题数从不超过 5-6，重复预测会抬高 Np、直接压低 precision。
只做检测与报告，不改文件；是否合并按报告里的证据重叠度判断。
"""
import argparse
import json
import re
import sys

sys.path.insert(0, "src")
from metric import read_jsonl, norm_issue, chain_text  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

enc = SentenceTransformer("models/bge-small-zh-v1.5", cache_folder="models", device="cpu")


def segset(it):
    return set(re.sub(r"\s+", "", s) for s in it["argument_chain"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--name-sim", type=float, default=0.80)
    args = ap.parse_args()

    rows = read_jsonl(args.pred)
    hits = []
    for r in rows:
        ils = r["issue_list"]
        names = [norm_issue(i["issue_name"]) for i in ils]
        vecs = enc.encode(names, normalize_embeddings=True)
        for a in range(len(ils)):
            for b in range(a + 1, len(ils)):
                sim = float(vecs[a] @ vecs[b])
                sa, sb = segset(ils[a]), segset(ils[b])
                jac = len(sa & sb) / max(1, len(sa | sb))
                same_stance = ils[a]["stance"] == ils[b]["stance"]
                if (sim >= args.name_sim and jac >= 0.3) or jac >= 0.8:
                    hits.append((r["sample_id"], round(sim, 3), round(jac, 2), same_stance,
                                 names[a], ils[a]["stance"], names[b], ils[b]["stance"]))
    print(f"疑似重复议题对: {len(hits)} 组 / {len(rows)} 篇")
    for h in sorted(hits, key=lambda x: -x[1])[:25]:
        sid, sim, jac, same, n1, s1, n2, s2 = h
        flag = "同立场" if same else "异立场(可能是真子议题,别合并)"
        print(f"  {sid} 名sim={sim:.2f} 证据重叠={jac:.2f} [{flag}]")
        print(f"     A {n1} ({s1})")
        print(f"     B {n2} ({s2})")
    with open(args.pred + ".duppairs.json", "w", encoding="utf-8") as fh:
        json.dump(hits, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
