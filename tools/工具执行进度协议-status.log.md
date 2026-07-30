# 工具执行进度协议 (status.log)

## 概述

`status.log` 采用 **JSON Lines** 格式（每行一条 JSON），工具在执行过程中追加写入，前端读取文件末尾即可获取最新状态。

文件位置：`{workspace}/logs/status.log`

兼容平台规约示例路径时，工具也会同步写入 `{workspace}/status.log`。

工具包同时输出：

- `{workspace}/logs/run.log`：人类可读的运行日志，包括启动信息、请求错误、查询进度和阶段耗时。
- `{workspace}/result/timing.json`：结构化阶段耗时和总耗时。
- `{workspace}/timing.json`：`result/timing.json` 的兼容副本。
- `{workspace}/intermediate/query_results.jsonl`：逐条查询结果 checkpoint。
- `{workspace}/intermediate/raw_tokens/`：可用于 `--continue` 的原始 token checkpoint。

## 消息类型

| type | 用途 | 何时写入 |
|------|------|----------|
| `total` | 声明总条数和阶段划分 | 执行开始时写一次 |
| `progress` | 汇报某阶段完成进度 | 每完成若干条目追加 |
| `completed` | 整个任务正常完成 | 全部阶段执行完毕 |
| `failed` | 整个任务异常终止 | 发生不可恢复错误 |

## 字段定义

```typescript
interface StatusLine {
  type: "total" | "progress" | "completed" | "failed"
  total_count: number          // 总条数
  stages: string[]             // 阶段列表，单阶段传 []
  stage?: string               // 当前阶段名（progress 时必填）
  finished_count?: number      // 当前阶段已完成条数（progress 时必填）
  error?: string               // 失败原因（failed 时必填）
}
```

## 示例

### 两阶段工具（query + eval 模型生成回复+裁判模型评估）

```jsonl
{"type":"total","total_count":100,"stages":["query","eval"]}
{"type":"progress","stage":"query","finished_count":30,"total_count":100}
{"type":"progress","stage":"query","finished_count":60,"total_count":100}
{"type":"progress","stage":"query","finished_count":100,"total_count":100}
{"type":"progress","stage":"eval","finished_count":20,"total_count":100}
{"type":"progress","stage":"eval","finished_count":50,"total_count":100}
{"type":"progress","stage":"eval","finished_count":100,"total_count":100}
{"type":"completed","total_count":100,"stages":["query","eval"]}
```

如只执行其中一个阶段，则stages只传要执行的阶段。如只进行评估："stages":["eval"]}

### 单阶段工具

```jsonl
{"type":"total","total_count":50,"stages":[]}
{"type":"progress","stage":"","finished_count":10,"total_count":50}
{"type":"progress","stage":"","finished_count":30,"total_count":50}
{"type":"progress","stage":"","finished_count":50,"total_count":50}
{"type":"completed","total_count":50,"stages":[]}
```

### 执行失败

```jsonl
{"type":"total","total_count":80,"stages":["query","eval"]}
{"type":"progress","stage":"query","finished_count":40,"total_count":80}
{"type":"failed","total_count":80,"stages":["query","eval"],"error":"模型服务连接超时"}
```

## 工具端写入规范

- 单次执行过程中只追加，不修改历史行；每次新执行开始时初始化新的 `total` 行
- 使用 类似`echo '{"type":"progress",...}' >> status.log` 方式追加写入
- `total` 行必须在第一行，执行开始时立即写入
- `completed` 或 `failed` 必须是最后一行，二选一
- 每个 `progress` 行都带 `total_count`，方便前端只读末尾即可计算
- 查询阶段的 `total_count` 包含稳定性复测任务；例如 `--limit 5` 通常对应 5 条唯一
  probe 和 7 个实际 query task，这是设计行为，不是重复执行错误。
- 冒烟测试使用 `--limit 1` 或 `--limit 5`；该参数限制查询任务规模，但不会阻止工具
  预先加载 reference bank 的索引和指纹文件。
- 运行失败时最后追加 `failed` 行；正常结束时最后追加 `completed` 行，二者不能同时出现。

## 调用方解析逻辑

1. 读取文件最后一行
2. 根据 `type` 判断状态：
   - `total` / `progress` → 运行中，展示进度
   - `completed` → 执行完成
   - `failed` → 执行失败，展示 error
3. 展示进度时：
   - 单阶段：`finished_count / total_count`
   - 两阶段：按 `stage` 分组取每个 stage 最后一条 progress，分别展示 `query: 30/100`、`eval: 50/100`
   - 整体进度 = 各 stage finished_count 之和 / (stage 数 × total_count)
