# p1-temporal-multifidelity-pde

Cost-matched multi-fidelity study of a time-dependent viscous Burgers surrogate.

Execution is on the remote Windows host only, inside `.venv` (PyTorch) and
`.venv-jax` (official Setinek grid code). `external/setinek` and
`external/pdebench` are unmodified upstream checkouts.

Level 1 is `REPRODUCTION_PARTIAL`. Level 2 is `COMPLETE`. Level 3 is
`NOT_TESTED` and is not authorized. The repository itself is `NO_COMMIT`.

## 文档索引（2026-10-26）

当前结论从这里读。10 月 23 日的报告只作历史记录，文件开头若有历史标记，则不能覆盖下表。

| 用途 | 文件 |
|---|---|
| 主张分级：已证实、有限支持、尚未验证 | `reports/18_CLAIM_LEDGER.md` |
| 执行摘要 | `reports/00_EXECUTIVE_SUMMARY.md` |
| 研究决定 | `reports/17_FINAL_RESEARCH_DECISION.md` |
| 论文体报告 | `reports/12_FINAL_PAPER_STYLE_REPORT.md` |
| 数值依据（种子均值、失败率范围） | `uploads/L2_ANALYSIS_PACKET.md` |
| 主表、各种子失败率、已转录配对区间 | `reports/15_L2_EXPERIMENT_RESULTS.md` |
| 生成、训练、推断、全生命周期、标准化 | `reports/16_COST_ACCOUNTING.md` |
| 求解器校准与三种轨迹对象 | `reports/14_SOLVER_CALIBRATION.md` |
| 图注 | `reports/FIGURE_CAPTIONS.md` |
| 版本 | `reports/VERSION_RECORD.md` |
| 测试记录 | `reports/TEST_REPORT.md` |
| 远程命令 | `reports/REPRODUCTION_COMMANDS.md` |
| 打开测试集之前的步数修订 | `reports/AMENDMENT_BEFORE_TEST.md` |
| Level 1 | `reports/07_L1_REPLICATION.md` |

历史状态文件：`reports/FINAL_RESEARCH_DECISION.md`，以及 `04`、`06`、`08`、`09`、`10`、`11`。它们描述的是数据尚未就绪或协议尚未执行时的记录。
