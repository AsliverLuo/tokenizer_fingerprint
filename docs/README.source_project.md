# Tokenizer Fingerprint BCS-only

这是从原始项目整理出的 BCS-only 版本，只保留 Boundary Consistency Score
相关流程：

1. 生成或加载 probe
2. 调用模型 API 获取 `max_tokens=1` 的 next-token 输出
3. 提取 token 边界签名
4. 计算模型间 BCS
5. 输出套壳/同源风险判定

## 目录

```text
tokenizer_fingerprint/
  cli.py                # generate-probes / build-reference / detect / compare
  detector.py           # BCS 检测主流程
  query_engine.py       # OpenAI-compatible / Anthropic 查询
  feature_extractor.py  # token 边界特征抽取
  similarity.py         # BCS 打分与判定
  reference_bank.py     # 参考库加载保存
  probe_generator.py    # 默认 probe 生成
  schema.py             # 数据结构

probes/
  default_probes.json   # 默认 500 条 BCS probe

reference_bank/
  probes_used.json      # 当前 BCS 参考库使用的 probe
  */*.json              # 各模型参考指纹

config.yaml             # 默认 BCS 配置
config_bcs.yaml         # BCS 配置副本
config_baichuan_bcs.yaml
config_deepseek_bcs.yaml
```

## 库内离线对比

如果只想在已有参考库内部比较模型，不调用 API、不生成新指纹，使用：

```bash
python -m tokenizer_fingerprint.cli compare-bank \
  --reference reference_bank/ \
  --output bank_compare.json \
  --csv-output bank_compare.csv \
  --top-k 5
```

## API 检测

```bash
python -m tokenizer_fingerprint.cli detect \
  --config config.yaml \
  --reference reference_bank/ \
  --probes reference_bank/probes_used.json \
  --output results_bcs/ \
  --concurrency 1
```

检测单个模型：

```bash
python -m tokenizer_fingerprint.cli detect \
  --config config.yaml \
  --target "Qwen3-32B" \
  --reference reference_bank/ \
  --probes reference_bank/probes_used.json \
  --output results_bcs/ \
  --concurrency 1
```

## 重建参考库

```bash
python -m tokenizer_fingerprint.cli build-reference \
  --config config.yaml \
  --probes reference_bank/probes_used.json \
  --output reference_bank_new/ \
  --concurrency 1
```

## 比较两个指纹

```bash
python -m tokenizer_fingerprint.cli compare \
  reference_bank/qwen/Qwen3-32B.json \
  reference_bank/qwen/Qwen3.5-397B-A17B.json
```

## Soft BCS 并行方法

为保留原始严格 BCS 流程，同时分析同家族 sibling 模型间的“软相似度”，项目新增了一套完全并行的 soft 版本：

```text
tokenizer_fingerprint/
  similarity.py        # 原始严格 BCS
  detector.py          # 原始检测主流程
  cli.py               # 原始 CLI

  similarity_soft.py   # 新增 Soft BCS
  detector_soft.py     # 新增 soft 检测流程
  cli_soft.py          # 新增 soft CLI
```

Soft BCS 不改变查询、probe、指纹提取和参考库格式，只改变相似度计算方式。  
它会在严格边界完全一致之外，给以下分项匹配部分分数：

- `exact`: 0.60
- `char_length`: 0.07
- `byte_length`: 0.07
- `token_type`: 0.16
- `prefix`: 0.07
- `empty`: 0.03

适用场景：

- 分析 `Qwen2.5-32B-Instruct` 与 `Qwen2.5-Coder-32B-Instruct` 这类同家族衍生模型
- 对比 sibling 模型的家族亲缘性，而不是只看严格 boundary exact match
- 不希望改动原始 `BCS-only` 流程

### Soft Detect

建议使用项目实际 conda 环境运行：

```bash
/home/jovyan/data/conda-envs/tokenizer/bin/python -m tokenizer_fingerprint.cli_soft detect \
  --config config.yaml \
  --target "Qwen2.5-32B-Instruct" \
  --reference reference_bank_core1000_en/ \
  --probes reference_bank_core1000_en/probes_used.json \
  --output results_soft_qwen25_32b/ \
  --concurrency 8
```

### Soft Compare

```bash
/home/jovyan/data/conda-envs/tokenizer/bin/python -m tokenizer_fingerprint.cli_soft compare \
  reference_bank_core1000_en/qwen/Qwen2.5-32B-Instruct.json \
  reference_bank_core1000_en/qwen/Qwen2.5-Coder-32B-Instruct.json
```

### Soft Compare-Bank

```bash
/home/jovyan/data/conda-envs/tokenizer/bin/python -m tokenizer_fingerprint.cli_soft compare-bank \
  --reference reference_bank_core1000_en/ \
  --output bank_compare_soft.json \
  --csv-output bank_compare_soft.csv \
  --top-k 5
```

说明：

- 原 `python -m tokenizer_fingerprint.cli ...` 完全不受影响
- 原 `similarity.py`、`detector.py`、`cli.py` 没有修改
- soft 版本只是并行实验入口，适合做家族相似性分析，不应直接替代严格 BCS 判定
