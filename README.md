# A股因子回测引擎

简体中文 | [English](README_EN.md)

[![CI](https://github.com/elexingyu/ashare-factor-backtest/actions/workflows/ci.yml/badge.svg)](https://github.com/elexingyu/ashare-factor-backtest/actions/workflows/ci.yml)
[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

一个面向 AI 和自动化研究流程的 A 股单因子回测引擎。输入表达式与数据配置，CLI 会完成安全编译、A 股交易约束处理、滚动检验，并输出稳定的机器可读结果。

> 本项目仅用于研究，不构成投资建议。回测结果和合成示例不能证明未来收益。

## 为什么做这个项目

通用回测框架通常把数据整理、股票池历史、复权、涨跌停、停牌和报告口径留给使用者处理。本项目把这些容易反复出错的 A 股规则固化为可复用协议，让人或 AI 只需提交类似 WorldQuant 风格的因子表达式，不必为每个实验重写一套回测代码。

它不是因子搜索器，也不承诺发现盈利因子。v0.1 专注于把一条已有表达式快速、可复现地变成一份可信的单因子回测结果。

## 快速开始

本项目可独立构建为 wheel。在当前目录运行：

```bash
uv sync --locked --all-groups
uv run ashare-backtest doctor --json
uv run ashare-backtest compile 'cs_rank(ts_pct_change(close,5))' --json
uv run ashare-backtest evaluate \
  --job examples/demo_daily/job.yaml \
  'cs_rank(ts_pct_change(close,5))' \
  --through rolling \
  --work-root /tmp/ashare-factor-demo \
  --json
```

CLI 每次只向标准输出写一行 JSON，适合由 Codex、Claude Code、流水线或其他程序直接调用。诊断日志写到标准错误，不会污染机器协议。

仓库自带的数据完全由程序生成，仅用于检查上市边界、ST 区间、停牌、开盘涨停和开盘跌停等执行规则，不用于展示 alpha。

## 核心能力

- **表达式编译：** 字段和算子采用白名单；拒绝未知字段、非法参数和前视引用。
- **A 股时间点语义：** 区分因子观察时点、股票池历史与下一开盘执行，避免使用未来成分股或未来行情。
- **复权与成交分离：** 因子计算可使用复权序列，交易可执行性与收益计算使用对应的真实价格语义。
- **交易约束：** 支持上市状态、ST、停牌、开盘涨跌停、费用和只做多组合。
- **滚动检验：** 输出训练段和测试段证据、策略与基准指标、超额指标及覆盖率。
- **可复现产物：** 表达式、数据身份、任务配置和评价语义共同决定产物身份，便于缓存、审计和复跑。
- **AI 友好接口：** `capabilities`、`schema`、`doctor`、`compile` 和 `evaluate` 均提供版本化 JSON 协议。

## 项目边界

v0.1 包含表达式求值、数据契约、单因子分组回测、A 股执行约束和滚动证据。

以下能力刻意不放进第一版：因子生成与搜索、多因子组合优化、实盘交易、私有行情数据和通用事件驱动订单系统。它们可以在上层调用本引擎，但不应混进单因子回测的可信计算核心。

## 性能证据

下面的表格由已归档 JSON 自动生成。对比固定了输入值、表达式语义、有效值掩码、Python 环境、进程数和输出范围；只有数值结果对齐后才允许显示速度比。

首轮对比测量的是两套引擎从各自原生持久化存储读取数据并生成四个因子矩阵的时间，不包含 A 股成交模拟、IC、费用和滚动检验。因此，这能证明表达式计算路径的当前性能，不能代表完整回测一定快同样倍数。

<!-- benchmark:start -->
**证据状态：** 可复现发布版本。

| 引擎 | 墙钟时间中位数 | 峰值内存 |
| --- | ---: | ---: |
| ashare-factor-backtest-public-evaluator | 0.086 s | 372 MiB |
| microsoft-qlib-local-provider-kernels-1 | 0.903 s | 690 MiB |

在该工作负载下，Qlib 与本引擎的墙钟时间比为 `10.50x`。工作负载包含 500 只证券、1500 个交易日和 4 条语义对齐的表达式。输出最大绝对误差为 `1.45e-05`，所有有限值位置完全一致。

该结果是在同一 Python 环境中，以单进程、缓存预热方式，从各自原生存储读取数据并生成因子矩阵。它不是完整回测速度对比，也不代表其他公式、数据规模或机器仍有相同比例。引用前请阅读 `BENCHMARKS.md` 和已归档的 JSON 结果。
<!-- benchmark:end -->

完整口径、原始 JSON 和复现命令见 [BENCHMARKS.md](BENCHMARKS.md)。

## 许可证

代码使用 Apache-2.0 许可证。合成示例数据单独以 CC0 1.0 发布，详见 [examples/DATA_LICENSE.md](examples/DATA_LICENSE.md)。
