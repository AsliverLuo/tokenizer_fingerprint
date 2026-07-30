# Tokenizer Fingerprint BCS-only 简化版（2026-07-30）

本目录是从开发仓库中整理出的最小可用版本：只保留最新使用的 500K 多语言语料/reference bank、最新必要源码、测试、生成脚本，以及按规约封装后的工具入口。

## 已保留内容

```text
.
├── tokenizer_fingerprint/      # 最新 Python 源码，支持 BCS / soft-BCS / next-token 查询
├── tools/                      # 规约工具包：入口 run.py / vocab.py + CPython 编译后的 .so 核心模块
├── reference_bank/             # 最新 500K 多语言 legacy-chat reference bank
├── probes/                     # 最新 500K 多语言 probe 与 smoke50 probe
├── scripts/                    # 生成 probe、启动本地模型、构建 reference bank 的必要脚本
├── configs/                    # 当前检测配置与最新本地 500K 运行配置
├── tests/                      # pytest 单元测试
└── docs/                       # 最新 500K 结果分析报告与原始项目 README
```

未保留历史实验目录、旧 reference bank、旧 probe、旧日志、缓存、图片和无关中间产物。

## 最新语料库与结果

- 最新语料 / probe：`probes/en_de_fr_ar_zh_integrated_500000_probes.json`
- smoke probe：`probes/en_de_fr_ar_zh_integrated_smoke50_probes.json`
- 最新 reference bank：`reference_bank/`
- 原始来源目录：`reference_bank_local_multilingual_500k_20260714_105014_legacy_chat`
- 运行配置快照：`configs/latest_local_multilingual_config.yaml`
- 结果分析报告：`docs/LOCAL_MULTILINGUAL_500K_ANALYSIS_REPORT.md`

该 reference bank 包含 11 个本地开源模型，每个模型 500,000 条 next-token 查询结果，总计 5,500,000 条 raw-token 记录。模型包括 Qwen2.5/Qwen3/Qwen3.5 与 Llama-3/Llama-3.1 系列。

> 为避免重复占用约 9GB 空间，`reference_bank/` 使用硬链接镜像复制；文件内容与原目录一致。若需要完全物理拷贝，可另行执行 `cp -a reference_bank <new_path>`。

## 标准采样协议

默认协议保持确定性 next-token 拟合：

```yaml
max_tokens: 1
temperature: 0
top_p: 1
presence_penalty: 0
frequency_penalty: 0
```

system prompt 见 `configs/latest_local_multilingual_config.yaml`。在线模型若不支持完全确定性采样，或会强制启用思考/安全模板/chat template，输出会受影响；此类差异应记录到 `api_config.extra_body` 与运行报告中。

## 常用命令

安装依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

运行单元测试：

```bash
python -m pytest tests/
```

离线比较 reference bank：

```bash
python -m tokenizer_fingerprint.cli compare-bank \
  --reference reference_bank/ \
  --output bank_compare.json \
  --csv-output bank_compare.csv \
  --top-k 5
```

使用已有 reference bank 检测目标模型：

```bash
python -m tokenizer_fingerprint.cli detect \
  --config configs/config_bcs.yaml \
  --reference reference_bank/ \
  --probes reference_bank/probes_used.json \
  --output results_bcs/ \
  --concurrency 1
```

生成/复现本地多语言 500K reference bank：

```bash
bash scripts/run_local_multilingual_nexttoken.sh
```

## 规约工具包入口

`tools/` 目录保留“入口 py + 核心 .so”的封装形式：

```bash
cd tools
python run.py --mode detect --model <model> --base_url <base_url> --api_key <api_key>
python run.py --mode vocab  --model <model> --base_url <base_url> --api_key <api_key>
python vocab.py --model <model> --base_url <base_url> --api_key <api_key>
```

`tools/assets/reference_bank` 已改为相对软链：`../../reference_bank`。

工具新增的目标模型采样超参数接口：

```bash
python run.py --mode vocab \
  --model <model> \
  --base_url <base_url> \
  --api_key <api_key> \
  --target_temperature 0 \
  --target_max_tokens 1 \
  --target_top_p 1 \
  --target_presence_penalty 0 \
  --target_frequency_penalty 0 \
  --target_extra_body_json '{"seed": 42}'
```

建议基准建库和检测均使用默认确定性参数；如果甲方要求测试不同温度/采样策略，应在输出报告中记录完整超参数。
