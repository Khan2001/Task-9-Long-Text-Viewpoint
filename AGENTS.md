# 项目约定

## 硬性规则：API 调用必须事先获得用户批准

本项目任何会消耗 token 的操作（目前指 `src/infer_api.py` 调用 DashScope 等在线 Qwen API），
执行前必须先用文字向用户报告以下三点，并等待用户明确批准（"可以""批准"等）：

1. 用途：跑哪个阶段、哪些数据、什么目的
2. 预估用量：调用次数、大致 token 量
3. 预估耗时

不得把"继续""下一步"之类的泛化回复当作 API 授权；每次 API 运行都要单独确认。

以下操作是本地零成本的，无需批准：`metric.py` 评测、`stance_calibrate.py` 后处理、
`retrieval.py` 检索、`pipeline.py` 拼装、`tune_post.py` 调参（复用 `runs/` 缓存）。

## 赛题与评测口径

- 赛题：谈判对手长文本观点挖掘与预测；`Score = 0.8 × F1_ext + 0.2 × S_pred`。
- 最终提交必须能在主办方指定的 Qwen3-32B 上离线复现；
  README 明确禁止用在线 API 直接生成最终 test 提交。
- 提交文件名为 `result.jsonl`，UTF-8 无 BOM，标准 JSONL。

## 工程约定

- 模型的原始输出缓存在 `runs/<run>/extract|future/*.json`；
  改后处理参数时优先复用缓存，不要重跑 API。
- 证据 `argument_chain` 必须是输入原文的逐字片段。
- 检索增强的范例来自 `data/train.jsonl`；训练集文档嵌入缓存在 `models/train_text_emb.npy`。
