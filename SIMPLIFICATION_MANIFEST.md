# 简化仓库清单

## 1. 选择依据

本简化版以当前开发仓库中最新完成且规模最大的多语言 reference bank 为基准：

```text
reference_bank_local_multilingual_500k_20260714_105014_legacy_chat
```

对应配置快照来自：

```text
vllm_logs/local_multilingual_20260714_105014_legacy_chat/local_multilingual_config.yaml
```

对应 probe 集来自：

```text
probes/en_de_fr_ar_zh_integrated_500000_probes.json
probes/en_de_fr_ar_zh_integrated_smoke50_probes.json
```

## 2. 源码范围

保留的源码范围：

- `tokenizer_fingerprint/*.py`：BCS / soft-BCS / reference-bank / query-engine 全部核心源码。
- `scripts/generate_multilingual_wikipedia_probes.py`：多语言 probe 生成。
- `scripts/run_local_multilingual_nexttoken.sh`：最新本地多语言 next-token reference bank 主运行脚本。
- `scripts/run_multilingual_minprompt_retry_500k_nohup.sh`：在线/重试版 500K 运行脚本。
- `scripts/start_local_qwen25_single.sh`、`scripts/start_local_llama_single.sh`：本地模型服务启动辅助脚本。
- `scripts/build_nexttoken_vocab_bank.py`：从 raw next-token 输出构建模拟词表/词表库。
- `scripts/build_tokenizer_baselines.py`、`scripts/download_tokenizers_hf.py`：开源 tokenizer baseline 相关辅助脚本。
- `tools/`：规约封装版本，保留 `run.py`、`vocab.py`、`start.sh`、配置、README、规约文档以及已编译 `.so`。

剔除的内容：历史 reference bank、旧实验输出、旧日志、缓存、无关图片、旧配置备份、非最新 probe 原始中间文件。

## 3. Reference bank 内容

`reference_bank/` 内包含：

- `index.json`：模型索引。
- `probes_used.json`：建库时实际使用的 500K probes。
- `raw_tokens/*.jsonl`：每个模型逐 probe 的 next-token 原始输出。
- `qwen/*.json`、`llama/*.json`：提取后的模型指纹。
- `bank_compare.json`、`bank_compare.csv`：库内两两 BCS 对比结果。

模型数量：11。每模型 probe 数：500,000。

## 4. 目录自检

可执行：

```bash
python -m json.tool reference_bank/index.json >/dev/null
python -m tokenizer_fingerprint.cli compare-bank --reference reference_bank --top-k 3
python -m pytest tests/
```

如使用 `tools/` 下已编译 `.so`，需匹配 CPython 3.12 / Linux x86_64。其他 Python 版本请从 `tokenizer_fingerprint/*.py` 重新编译。
