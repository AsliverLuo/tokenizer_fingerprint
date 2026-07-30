# 本地多语言 500K BCS 结果分析报告

## 运行概况

结果目录：

```text
reference_bank_local_multilingual_500k_20260714_105014
```

使用的 probe 集：

```text
probes/en_de_fr_ar_zh_integrated_500000_probes.json
```

本次 reference bank 共完成 11 个模型、55 组两两比较。每个已完成模型都有 500,000 条对齐的多语言 next-token 查询结果，其中英语、德语、法语、阿拉伯语、中文各 100,000 条。

已完成的 raw-token 文件中没有 API error。

## 已完成模型

| 模型 | 家族 | Probe 数 |
|---|---:|---:|
| Qwen2.5-7B-Local | qwen | 500,000 |
| Qwen2.5-7B-Instruct-Local | qwen | 500,000 |
| Qwen3-1.7B-Local | qwen | 500,000 |
| Qwen3-8B-Local | qwen | 500,000 |
| Qwen3-8B-Base-Local | qwen | 500,000 |
| Qwen3.5-9B-Local | qwen | 500,000 |
| Qwen3.5-9B-Base-Local | qwen | 500,000 |
| Meta-Llama-3-8B-Local | llama | 500,000 |
| Meta-Llama-3-8B-Instruct-Local | llama | 500,000 |
| Meta-Llama-3.1-8B-Local | llama | 500,000 |
| Meta-Llama-3.1-8B-Instruct-Local | llama | 500,000 |

## Raw 输出质量

Qwen 系列整体空输出率较低。Llama instruct 模型可用，但中文 probe 上空输出较多。两个 Llama base 模型空输出率明显偏高，应作为低置信度 reference 使用。

| 模型 | 空输出数 | 空输出率 | 主要集中语言 |
|---|---:|---:|---|
| Qwen2.5-7B-Instruct-Local | 400 | 0.0800% | 中文 |
| Qwen3-8B-Base-Local | 468 | 0.0936% | 中文 |
| Qwen3-8B-Local | 564 | 0.1128% | 中文 |
| Qwen3.5-9B-Local | 696 | 0.1392% | 中文 |
| Qwen3-1.7B-Local | 745 | 0.1490% | 中文 |
| Qwen3.5-9B-Base-Local | 820 | 0.1640% | 中文 |
| Qwen2.5-7B-Local | 1,226 | 0.2452% | 中文 |
| Meta-Llama-3-8B-Instruct-Local | 2,945 | 0.5890% | 中文 |
| Meta-Llama-3.1-8B-Instruct-Local | 3,179 | 0.6358% | 中文 |
| Meta-Llama-3.1-8B-Local | 31,997 | 6.3994% | 多语言混合，英语/法语/中文较多 |
| Meta-Llama-3-8B-Local | 35,380 | 7.0760% | 多语言混合，英语/法语/中文较多 |

有两个模型出现明显的协议或 chat template 污染：

| 模型 | 异常现象 |
|---|---|
| Qwen2.5-7B-Local | `assistant` 在高频输出中出现约 25k 次 |
| Qwen3-8B-Base-Local | `Assistant` 和 `user` 合计出现约 193k 次 |

这些现象说明部分 base 模型通过 chat completions 查询时，角色文本泄漏到了 next-token 输出中。这会扭曲 BCS 分数，使结果不再纯粹反映 tokenizer 或 next-token 边界行为。

## 各模型高频输出

下表列出每个模型 raw token 结果中出现频率最高的 10 个输出。比例按 500,000 条 probe 计算。`<EMPTY>` 表示空输出。

### Qwen 系列

| 模型 | Top-10 高频输出 |
|---|---|
| Qwen2.5-7B-Instruct-Local | `ال` 26,716 (5.3432%); `t` 7,961 (1.5922%); `s` 7,685 (1.5370%); `n` 5,515 (1.1030%); `的` 5,267 (1.0534%); `tion` 4,102 (0.8204%); `ion` 3,456 (0.6912%); `r` 3,337 (0.6674%); `e` 3,127 (0.6254%); `l` 3,096 (0.6192%) |
| Qwen2.5-7B-Local | `assistant` 25,439 (5.0878%); `ال` 22,424 (4.4848%); `n` 13,792 (2.7584%); `The` 12,362 (2.4724%); `1` 11,156 (2.2312%); `s` 9,293 (1.8586%); `Assistant` 7,834 (1.5668%); `de` 5,988 (1.1976%); `Die` 5,827 (1.1654%); `tion` 4,529 (0.9058%) |
| Qwen3-1.7B-Local | `ال` 24,176 (4.8352%); `1` 7,920 (1.5840%); `的` 7,833 (1.5666%); `de` 6,664 (1.3328%); `s` 5,814 (1.1628%); `n` 5,326 (1.0652%); `The` 4,701 (0.9402%); `d` 4,552 (0.9104%); `the` 4,416 (0.8832%); `أ` 4,298 (0.8596%) |
| Qwen3-8B-Local | `s` 11,056 (2.2112%); `t` 10,710 (2.1420%); `,` 9,499 (1.8998%); `ة` 7,266 (1.4532%); `e` 6,277 (1.2554%); `n` 6,206 (1.2412%); `ي` 5,853 (1.1706%); `的` 5,837 (1.1674%); `r` 5,594 (1.1188%); `en` 5,039 (1.0078%) |
| Qwen3-8B-Base-Local | `Assistant` 96,797 (19.3594%); `user` 96,130 (19.2260%); `ال` 42,930 (8.5860%); `The` 25,487 (5.0974%); `1` 23,602 (4.7204%); `the` 12,078 (2.4156%); `Die` 9,034 (1.8068%); `der` 8,492 (1.6984%); `,` 8,121 (1.6242%); `2` 7,103 (1.4206%) |
| Qwen3.5-9B-Local | `ت` 6,002 (1.2004%); `t` 5,955 (1.1910%); `s` 5,413 (1.0826%); `n` 5,221 (1.0442%); `ال` 4,962 (0.9924%); `ي` 4,553 (0.9106%); `的` 4,031 (0.8062%); `م` 3,887 (0.7774%); `r` 3,707 (0.7414%); `tion` 3,543 (0.7086%) |
| Qwen3.5-9B-Base-Local | `ال` 9,083 (1.8166%); `ت` 6,262 (1.2524%); `م` 5,543 (1.1086%); `n` 5,503 (1.1006%); `t` 5,273 (1.0546%); `s` 4,982 (0.9964%); `The` 4,979 (0.9958%); `ي` 4,409 (0.8818%); `ل` 4,009 (0.8018%); `的` 3,901 (0.7802%) |

### Llama 系列

| 模型 | Top-10 高频输出 |
|---|---|
| Meta-Llama-3-8B-Instruct-Local | `ال` 42,241 (8.4482%); `t` 11,709 (2.3418%); `的` 8,993 (1.7986%); `م` 7,497 (1.4994%); `n` 7,159 (1.4318%); `de` 6,954 (1.3908%); `في` 5,751 (1.1502%); `l` 5,359 (1.0718%); `d` 4,221 (0.8442%); `r` 4,052 (0.8104%) |
| Meta-Llama-3-8B-Local | `<EMPTY>` 35,380 (7.0760%); `-` 17,528 (3.5056%); `,` 5,452 (1.0904%); `的` 5,077 (1.0154%); `...\n` 4,703 (0.9406%); `­` 4,519 (0.9038%); `.` 3,760 (0.7520%); `ية` 3,610 (0.7220%); `ين` 1,785 (0.3570%); `en` 1,679 (0.3358%) |
| Meta-Llama-3.1-8B-Instruct-Local | `ال` 13,384 (2.6768%); `t` 10,568 (2.1136%); `的` 7,509 (1.5018%); `n` 6,267 (1.2534%); `م` 5,941 (1.1882%); `s` 5,559 (1.1118%); `أ` 5,519 (1.1038%); `r` 4,502 (0.9004%); `h` 4,348 (0.8696%); `d` 4,134 (0.8268%) |
| Meta-Llama-3.1-8B-Local | `<EMPTY>` 31,997 (6.3994%); `-` 19,247 (3.8494%); `­` 5,723 (1.1446%); `的` 5,545 (1.1090%); `,` 5,390 (1.0780%); `.` 3,726 (0.7452%); `...\n` 3,594 (0.7188%); `ية` 3,429 (0.6858%); `é` 1,710 (0.3420%); `ين` 1,664 (0.3328%) |

从高频输出看，污染最严重的是 `Qwen3-8B-Base-Local`：`Assistant` 和 `user` 两个角色词合计占比约 38.59%。`Qwen2.5-7B-Local` 也有明显角色词泄漏，`assistant` 与 `Assistant` 合计约 6.65%。两个 Llama base 模型的首位高频输出均为 `<EMPTY>`，与前面的空输出率结论一致。

## 两两 BCS 结果

最高分 pair 如下：

| 排名 | 模型对 | BCS |
|---:|---|---:|
| 1 | Meta-Llama-3-8B-Local vs Meta-Llama-3.1-8B-Local | 0.832506 |
| 2 | Qwen3.5-9B-Local vs Qwen3.5-9B-Base-Local | 0.717328 |
| 3 | Qwen2.5-7B-Local vs Qwen3-1.7B-Local | 0.673198 |
| 4 | Meta-Llama-3-8B-Instruct-Local vs Meta-Llama-3.1-8B-Instruct-Local | 0.574846 |
| 5 | Qwen3-8B-Local vs Qwen3.5-9B-Local | 0.532546 |
| 6 | Qwen2.5-7B-Instruct-Local vs Qwen3-8B-Local | 0.526348 |
| 7 | Qwen3-8B-Local vs Meta-Llama-3.1-8B-Instruct-Local | 0.525398 |
| 8 | Qwen3-8B-Local vs Qwen3.5-9B-Base-Local | 0.522470 |
| 9 | Qwen3.5-9B-Base-Local vs Meta-Llama-3.1-8B-Instruct-Local | 0.520016 |
| 10 | Qwen3.5-9B-Local vs Meta-Llama-3.1-8B-Instruct-Local | 0.517436 |

最容易解释的强匹配关系包括：

- Llama-3 base 与 Llama-3.1 base；
- Qwen3.5-9B 与 Qwen3.5-9B-Base；
- Qwen2.5-7B 与 Qwen3-1.7B；
- Llama-3 instruct 与 Llama-3.1 instruct。

## 最近邻可信度

`compare-bank` 使用的 same-source 判定规则为：

```json
{
  "top1_score_threshold": 0.5,
  "top1_minus_top2_threshold": 0.08
}
```

满足该规则的匹配如下：

| 模型 | 最近邻 | 分数 | Top1-Top2 差距 |
|---|---|---:|---:|
| Qwen2.5-7B-Local | Qwen3-1.7B-Local | 0.673198 | 0.341470 |
| Qwen3-1.7B-Local | Qwen2.5-7B-Local | 0.673198 | 0.297550 |
| Qwen3.5-9B-Local | Qwen3.5-9B-Base-Local | 0.717328 | 0.184782 |
| Qwen3.5-9B-Base-Local | Qwen3.5-9B-Local | 0.717328 | 0.194858 |
| Meta-Llama-3-8B-Local | Meta-Llama-3.1-8B-Local | 0.832506 | 0.477984 |
| Meta-Llama-3.1-8B-Local | Meta-Llama-3-8B-Local | 0.832506 | 0.478116 |
| Meta-Llama-3-8B-Instruct-Local | Meta-Llama-3.1-8B-Instruct-Local | 0.574846 | 0.132024 |

不稳定或不应强判的最近邻如下：

| 模型 | Top1 | 分数 | 差距 | 原因 |
|---|---|---:|---:|---|
| Qwen2.5-7B-Instruct-Local | Qwen3-8B-Local | 0.526348 | 0.012348 | Top1 和 Top2 过近 |
| Qwen3-8B-Local | Qwen3.5-9B-Local | 0.532546 | 0.006198 | 候选几乎并列 |
| Meta-Llama-3.1-8B-Instruct-Local | Meta-Llama-3-8B-Instruct-Local | 0.574846 | 0.049448 | 跨家族第二名过近 |
| Qwen3-8B-Base-Local | Qwen2.5-7B-Local | 0.319670 | 0.018006 | 分数低且存在协议污染 |

## 家族级区分能力

同家族和跨家族分数分布存在明显重叠：

| Pair 类型 | 数量 | 最小值 | 中位数 | 平均值 | 最大值 |
|---|---:|---:|---:|---:|---:|
| 同家族 | 27 | 0.163346 | 0.319670 | 0.389661 | 0.832506 |
| 跨家族 | 28 | 0.147056 | 0.326873 | 0.329208 | 0.525398 |

这说明在当前 bank 上，不能只靠一个全局 BCS 阈值做家族分类。有些跨家族 pair 的分数超过 0.5，而有些同家族 pair 因 endpoint 或 prompt 协议影响分数很低。

最高跨家族 pair 为：

```text
Qwen3-8B-Local vs Meta-Llama-3.1-8B-Instruct-Local: 0.525398
```

因此，本批结果更适合用于发现强相似 pair 和分析局部邻近关系，不适合直接作为最终的家族分类 benchmark。

## 关键结论

1. 本次 full bank 基本完整：11 个模型均有 500k 条结果，语言分布均衡，API error 为 0。

2. BCS 成功恢复了一些预期强关系，包括 Llama base 之间、Qwen3.5 9B/base 之间、Llama instruct 之间。

3. 最大噪声来源是 base 模型的查询协议。`Qwen3-8B-Base-Local` 和 `Qwen2.5-7B-Local` 出现角色词泄漏；两个 Llama base 模型空输出率偏高。

4. 同家族和跨家族 BCS 分布重叠明显。实际判断时，最近邻差距比单个分数更可靠。

5. `Qwen3-4B-Instruct-2507-Local` 缺失，如果需要完整覆盖本地模型，应单独补跑并重新生成 `bank_compare`。

## 后续建议

1. 优先将 base 模型改为 completions 风格的接口或纯 prompt 查询方式后重跑：
   - `Qwen3-8B-Base-Local`
   - `Qwen2.5-7B-Local`
   - `Meta-Llama-3-8B-Local`
   - `Meta-Llama-3.1-8B-Local`

2. 当前 bank 可用于探索性分析和强邻居证据，但不建议直接作为最终家族分类基准。

3. 增加协议污染诊断指标，例如统计 `assistant`、`Assistant`、`user` 等角色词输出频率。

4. 补跑缺失的 `Qwen3-4B-Instruct-2507-Local`，再重新运行 `compare-bank`。

5. 后续检测决策建议采用组合规则：
   - Top1 BCS 足够高；
   - Top1-Top2 差距足够大；
   - 空输出率低；
   - 无角色词泄漏；
   - 在各语言子集上结论一致。
