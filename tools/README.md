# Tokenizer Fingerprint BCS Tool

本工具包用于在测评平台中运行 strict Boundary Consistency Score (BCS)
tokenizer 指纹检测。工具面向 OpenAI-compatible 目标模型服务，通过单 token
续写结果抽取 tokenizer 边界指纹，并与内置 reference bank 做同源性判定。

## 封装形态

当前工具包按平台规约采用以下交付形态：

- 入口文件保留为 Python 脚本：`run.py`。
- 核心指纹计算模块已封装为 CPython 扩展模块：`tokenizer_fingerprint/*.so`。
- 默认探针与参考指纹库入口为：`assets/reference_bank/`。
- 参考库已随交付包配置，实际内容以交付版本为准。
- 平台配置文件为：`settings.json`。
- 依赖声明文件为：`requirements.txt`。
- 进度协议说明为：`工具执行进度协议-status.log.md`。

入口 `run.py` 负责平台参数解析、workspace 初始化、日志与 status 写入、
断点续跑、结果落盘和调用编译后的核心模块。业务核心流程由
`tokenizer_fingerprint` 编译模块提供。

## 运行环境

`.so` 文件名为 `*.cpython-312-x86_64-linux-gnu.so`，因此运行环境需要满足：

- Linux x86_64
- CPython 3.12
- 安装 `requirements.txt` 中的运行依赖

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

当前交付包适配 CPython 3.12。若运行环境版本不同，请联系交付方提供对应运行包。

## Quick Start

### 目标模型服务

运行检测前，请准备兼容 OpenAI API 的目标模型服务，并确认服务支持
`/v1/chat/completions` 或 `/v1/completions` 接口，以及 `max_tokens=1` 请求参数。
目标服务的地址、鉴权信息和模型名称由使用方根据实际部署情况填写。

### BCS 检测

从工具包根目录运行：

```bash
sh start.sh \
  --mode detect \
  --workspace /path/to/workspace \
  --reference assets/reference_bank \
  --target_base_url http://127.0.0.1:8000/v1 \
  --target_api_key <API_KEY> \
  --target_model <MODEL_NAME> \
  --limit 5 \
  --parallel_num 1
```

上面的命令是冒烟测试，不是全量测试。`--limit 5` 最多选择 5 条唯一 probe；由于
工具默认抽取 10% probe 做 3 次稳定性复测，实际查询任务数可能是 7 条。若只做最快的
接口连通性验证，可将 `--limit` 改为 `1`。

测试完成后检查：

```bash
cat /path/to/workspace/result/result.json
cat /path/to/workspace/result/timing.json
tail -n 30 /path/to/workspace/logs/run.log
```

成功时 `result/result.json` 的 `status` 为 `completed`。

默认 reference bank 为 `assets/reference_bank/`，默认 probe 文件为
`assets/reference_bank/probes_used.json`。

注意：请使用 `sh start.sh` 或 `python3 run.py` 作为工具入口。

### Next-Token 模拟词表

`vocab` 模式输入目标模型名称、base URL 和 API key，对 probe 集逐条请求
`max_tokens=1` 的 next token，并输出两类结果：

- raw next-token JSONL，字段形式与历史 `raw_tokens/*.jsonl` 保持一致。
- 按 `output_text` 聚合的模拟词表频次文件。

示例：

```bash
python3 run.py \
  --mode vocab \
  --workspace /tmp/tkfp_vocab \
  --target_base_url http://127.0.0.1:18001/v1 \
  --target_api_key local-test-key \
  --target_model Qwen/Qwen2.5-7B \
  --datasets /abs/path/to/wikitext_truncated_100000_probes.json \
  --parallel_num 8 \
  --auto_save_batch_size 1000
```

如果不传 `--datasets`，工具会使用 `assets/reference_bank/probes_used.json`。做冒烟
测试时建议配合 `--limit`；若要拟合其他数据分布下的模拟词表，可以传入自定义 probe JSON。

## 平台参数

- `--mode`: `detect` 或 `vocab`。默认 `detect`。`vocab` 用于 next-token
  模拟词表抽取。
- `--workspace`: 输出工作目录，用于保存日志、中间结果和最终结果。
- `--datasets PATH [PATH ...]`: 绝对路径 probe JSON 文件。多个文件会按 `id`
  合并去重。
- `--limit N`: 仅运行加载后的前 `N` 条 probes，常用于冒烟测试。
- `--continue`: 兼容平台续跑参数。工具会复用 workspace 中已有 raw token
  checkpoint。
- `--parallel_num N`: 目标模型请求并发数。
- `--target_base_url`: OpenAI-compatible API base URL。
- `--target_api_key`: 目标模型 API key。本地服务可传任意非空值。
- `--target_model`: 发送给 API 的目标模型名称。
- `--target_temperature`: 目标模型采样温度。默认 `0`。
- `--target_max_tokens`: 覆盖 `max_tokens`。默认 `1`。
- `--target_top_p`: 覆盖 `top_p`。默认 `1`。
- `--target_presence_penalty`: 覆盖 `presence_penalty`。默认 `0`。
- `--target_frequency_penalty`: 覆盖 `frequency_penalty`。默认 `0`。
- `--target_extra_body_json`: 额外请求体 JSON 对象，用于传入后端特有超参数，
  例如 `seed`、`stop`、`enable_thinking` 等。
- `--reference`: 可选 reference bank 目录。默认使用内置
  `assets/reference_bank/`。
- `--auto_save_batch_size`: 每完成多少条查询任务刷新一次 raw token checkpoint。
- `--attack_*`: 兼容平台参数，当前 single-stage BCS 工具不使用。
- `--eval_*`: 兼容平台参数，当前 single-stage BCS 工具不使用。
- `--eval_scope`: 支持 `test` 和 `full`。`eval` 不适用于 BCS 单阶段流程。

## 检测流程

1. 加载 probes。若传入 `--datasets`，先合并并去重；否则使用 reference bank
   中的 `probes_used.json`。
2. 随机抽取 10% probes 做稳定性复测，默认重复 3 次。
3. 使用 OpenAI-compatible 接口查询目标模型，默认请求参数为
   `max_tokens=1`、`temperature=0`、`top_p=1`；如显式传入超参数接口，则按传入值覆盖。
4. 将每条输出转换为 `SingleTokenResult`，记录字符长度、字节长度、前导空格、
   前导换行、token 类型、是否为空等边界信息。
5. 抽取目标模型 `ModelFingerprint`。
6. 加载 reference bank，按 `probe_id` 对齐目标与参考模型的原始单 token 结果。
7. 计算 strict BCS：边界签名完全一致的 probe 数 / 共同 probe 数。
8. 结合 cross-family baseline、top1-top2 margin、bootstrap 方差、family
   consistency 和稳定性方差生成最终判定。

## Reference Bank 说明

Reference bank 是工具进行同源性判定所需的参考数据，默认入口为
`assets/reference_bank/`。交付和部署时，应确保该目录及其实际指向的文件完整可用；
如果使用符号链接，应在目标环境重新配置有效链接。

正式检测应使用与当前检测协议和目标模型匹配的 reference bank。除非交付方另有说明，
使用方无需修改或重新生成该目录内容。

reference bank 的标准结构如下：

```text
assets/reference_bank/
├── index.json
├── probes_used.json
├── bank_compare.json              # 可选，参考库内两两比较结果
├── bank_compare.csv               # 可选
├── raw_tokens/
│   └── <model>.jsonl              # 每条 probe 的 next-token 原始结果
├── qwen/
│   └── <model>.json               # 模型指纹 JSON
└── llama/
    └── <model>.json
```

其中必须包含：

- `index.json`: reference bank 索引，列出模型名、family、JSON 路径和 probe 数。
- `probes_used.json`: 构建该 reference bank 时使用的 probe 集。
- `<family>/<model>.json`: 每个开源模型的指纹文件，通常内含 `raw_results`。

Reference bank 的新增、替换和版本管理由交付方统一完成。使用方如需变更参考模型或
探针集，应先确认参考库与检测协议保持一致，再进行版本更新。

## 采样参数与温度配置

BCS 和 next-token 模拟词表都依赖“同一 probe 上的单 token 边界输出”。为了让不同
模型之间可比较，工具默认使用确定性 next-token 查询协议：

```text
max_tokens = 1
temperature = 0
top_p = 1
presence_penalty = 0
frequency_penalty = 0
```

这些默认值来自 `tokenizer_fingerprint.query_engine.DEFAULT_QUERY_PARAMS`。
工具同时提供显式超参数接口；只要用户传入对应参数，就会覆盖默认请求体：

```bash
python3 run.py \
  --mode detect \
  --workspace /tmp/tkfp_hp \
  --target_base_url http://127.0.0.1:18001/v1 \
  --target_api_key local-test-key \
  --target_model Qwen/Qwen2.5-7B \
  --limit 100 \
  --target_temperature 0 \
  --target_max_tokens 1 \
  --target_top_p 1 \
  --target_presence_penalty 0 \
  --target_frequency_penalty 0
```

后端特有参数可通过 JSON 传入：

```bash
python3 run.py \
  --mode vocab \
  --workspace /tmp/tkfp_vocab_hp \
  --target_base_url http://127.0.0.1:18001/v1 \
  --target_api_key local-test-key \
  --target_model Qwen/Qwen3-8B \
  --limit 100 \
  --target_extra_body_json '{"seed": 42, "enable_thinking": false}'
```

实际使用的超参数会写入：

- `result/result.json` 的 `request_hyperparameters`
- `result/summary.json` 的 `request_hyperparameters`
- `result/simulated_vocab.json` 的 metadata（仅 `vocab` 模式）

固定温度的原因：

- `temperature=0` 尽量使用贪心解码，降低随机采样噪声。
- BCS 比较的是 tokenizer 边界特征，不是模型创造性，因此不需要高温采样。
- reference bank 与目标模型必须使用同一套采样协议，否则分数不可比。

因此，正式 BCS 检测建议保持默认超参数。只有在甲方明确要求复现实验条件、验证温度
敏感性，或对某个后端必须传入特定参数时，才建议使用超参数覆盖接口。若检测目标使用了
非默认温度或 `max_tokens > 1`，对应 reference bank 也应使用完全相同的超参数重新制作。

不同模型服务的接口形态需要按模型配置：

- base model 或没有 chat template 的服务：优先使用 `endpoint: completions`。
- chat / instruct 模型：通常使用 `endpoint: chat_completions`。
- 需要 assistant 前缀续写的服务：可使用 `message_mode: assistant_prefill`。
- DeepSeek beta prefix completion 类接口：可使用 `message_mode: deepseek_chat_prefix`。
- 本地 Qwen thinking 模型：工具会自动加入 `enable_thinking=false` 相关参数，避免输出
  thinking marker 污染 next-token 结果。

## 在线模型适用边界

该工具适用于“能稳定返回单 token 续写”的在线模型服务，但不能保证所有在线模型都适用。
最低要求如下：

- 服务兼容 OpenAI `/v1/chat/completions` 或 `/v1/completions` 协议。
- 支持 `max_tokens=1`。
- 支持或至少接受 `temperature=0`、`top_p=1`。
- 返回内容应是直接续写文本，而不是解释、拒答、结构化 wrapper 或 reasoning 过程。
- 同一模型构建 reference bank 和检测目标时，应使用同一 endpoint、prompt 模式和采样参数。

可能不适用或需要适配的情况：

- 服务不支持 `temperature=0`，或实际不按贪心解码执行。
- 服务强制加入系统模板、免责声明、思考过程、JSON wrapper 等额外内容。
- 服务不支持 `max_tokens=1`，或者会把 token 合并/截断成非预期文本。
- 模型开启 reasoning/thinking，导致第一个 token 是 `<think>`、`Thinking` 等模板内容。
- 在线 API 后端版本频繁变化，导致同一模型在不同日期输出不一致。

遇到上述情况时，应先做小规模 `--limit` 冒烟测试，检查：

- raw token JSONL 中的 `output_text` 是否是纯续写。
- `empty_output_rate` 是否异常偏高。
- 是否大量出现模板词、解释性文本或 thinking marker。
- 同一批 probes 重跑后输出是否稳定。

只有冒烟测试通过后，才建议把该在线模型纳入正式 reference bank 或正式检测。

## 模拟词表流程

当前工具原本已经在 BCS/reference-bank 流程中执行 next-token 查询：默认查询协议为
`max_tokens=1`、`temperature=0`，并把每条 probe 的结果写入 raw token JSONL。
新增的 `--mode vocab` 把这部分能力独立出来，不再执行 reference bank 相似度检测，
而是直接生成模拟词表。

`vocab` 模式流程：

1. 加载 probes。推荐使用 WikiText 词内截断 probe 集；也可以用 `--limit`
   做小规模冒烟测试。
2. 用 OpenAI-compatible API 查询目标模型的 next token。
3. 写出 raw JSONL，每行字段包括 `probe_id`、`model_name`、`output_text`、
   `char_length`、`byte_length`、`has_leading_space`、`has_leading_newline`、
   `token_type`、`is_empty`、`latency_ms`。
4. 按 `output_text` 聚合频次，得到纯 next-token 采样拟合出的模拟词表。

## 输出文件

对于 `--workspace /path/to/ws`，`detect` 模式会写入：

- `/path/to/ws/logs/run.log`
- `/path/to/ws/logs/status.log`
- `/path/to/ws/status.log`，与 `logs/status.log` 同步写入，兼容平台规约示例路径
- `/path/to/ws/config.json`
- `/path/to/ws/intermediate/selected_probes.json`
- `/path/to/ws/intermediate/merged_probes.json`，仅当传入 `--datasets` 时生成
- `/path/to/ws/intermediate/query_results.jsonl`
- `/path/to/ws/intermediate/raw_tokens/<target_model>.jsonl`
- `/path/to/ws/result/evaluation_results.jsonl`
- `/path/to/ws/result/timing.json`
- `/path/to/ws/result/result.json`
- `/path/to/ws/result/target_fingerprint.json`
- `/path/to/ws/result/summary.json`
- `/path/to/ws/timing.json`
- `/path/to/ws/summary.json`

`vocab` 模式会写入：

- `/path/to/ws/logs/run.log`
- `/path/to/ws/logs/status.log`
- `/path/to/ws/status.log`，与 `logs/status.log` 同步写入，兼容平台规约示例路径
- `/path/to/ws/config.json`
- `/path/to/ws/intermediate/selected_probes.json`
- `/path/to/ws/intermediate/query_results.jsonl`
- `/path/to/ws/result/raw_tokens/<target_model>.jsonl`
- `/path/to/ws/result/<target_model>_raw_tokens.jsonl`
- `/path/to/ws/result/evaluation_results.jsonl`
- `/path/to/ws/result/simulated_vocab.csv`
- `/path/to/ws/result/simulated_vocab.json`
- `/path/to/ws/result/timing.json`
- `/path/to/ws/result/result.json`
- `/path/to/ws/result/summary.json`
- `/path/to/ws/timing.json`
- `/path/to/ws/summary.json`

`status.log` 使用 JSON Lines 格式。单次执行开始时写入新的 `total` 行，
执行过程中按规约追加写入：

- `total`: 声明总任务数
- `progress`: 查询进度
- `completed`: 正常完成
- `failed`: 异常终止

运行日志 `logs/run.log` 记录初始化信息、请求失败原因、查询进度、各阶段耗时和总耗时。
耗时结构化结果写入 `result/timing.json`，并同步写入 workspace 根目录的 `timing.json`。
查询 checkpoint 写入 `intermediate/raw_tokens/` 和 `intermediate/query_results.jsonl`，
可通过 `--continue` 复用已完成任务。

## 结果语义

`result/result.json` 中的关键字段：

- `label`: `same_source` 或 `not_same_source`
- `confidence`: 判定置信度
- `same_source_of`: 若判定同源，给出最接近的参考模型
- `top_matches`: BCS 排名前若干的参考模型
- `evidence`: z-score、margin sigma、family consistency 等证据
- `diagnosis`: 人类可读诊断理由
- `target_fingerprint`: 目标模型 tokenizer 指纹

`vocab` 模式的 `simulated_vocab.csv` 字段：

- `model_name`: 目标模型名
- `next_token_json`: JSON 转义后的 next token 文本
- `frequency`: 该 token 在采样结果中的出现次数
- `share`: 该 token 的频率占比
- `rank`: 按频次降序排序的名次
- `char_length`: 字符长度
- `byte_length`: UTF-8 字节长度

## 当前边界

- 当前平台入口只运行 strict BCS。soft BCS 模块保留在包内，但 `run.py`
  未将其作为默认入口。
- 入口 `run.py` 仍包含平台适配与执行编排逻辑；核心 tokenizer 指纹相关模块
  已编译为 `.so`。
- `vocab` 模式只基于 API 返回的 next-token 文本做频次拟合，无法读取真实 tokenizer
  的完整内部词表；它输出的是 next-token 采样模拟词表。
- 本工具需要目标模型服务兼容 `/v1/chat/completions`，部分 base model 或无 chat
  template 的服务会自动回退到 `/v1/completions`。
- 工具不会使用 `attack_*` 或 `eval_*` 参数，它们仅为平台统一参数兼容保留。
