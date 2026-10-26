# Phase B5 — R2.16 决策增补记录（Supplemental Decision Record）

**日期**: 2026-10-26
**类型**: 增补记录（Supplement）— **不覆盖、不重写**任何历史决策文件
**基线提交**: `d4fb934f6b9bf337b87405f3b23e9fe4c69febaa`（B5 实施）

---

## 一、文档目的与边界

本记录是 R2.15（P1–P8 确认清单）之后的增补，用于：

1. 将 2026-10-26 用户对 P1–P8 的口头/指令批准与实施事实归档；
2. 记录 P5 字段数量与旧文档的冲突及裁决；
3. 明确区分"用户决策已确认 / 代码已实施 / 测试已验证 / 真实训练尚未验证"四种状态；
4. 标注代码实施中发现并修复的 1 处旧行为偏差（`trainer.py` 日志键一致性）。

**本文件不授权任何新操作**；真实 B5 训练、评估与 checkpoint 写入仍未授权。

---

## 二、P1–P8 批准状态变更（2026-10-26）

历史状态（`B5_Final_Decisions_Confirmed.md` R2.14、`B5_Unresolved_Rules_Proposals.md` R2.13）中 P1–P8 全部为 `PROPOSED — NOT APPROVED`。

2026-10-26，用户通过授权指令将 P1–P8 作为**最新用户决定**纳入 B5 实施范围。本记录将 P1–P8 状态更新为：

| ID | 提案 | 旧状态（R2.13/R2.14） | 新状态（R2.16，2026-10-26） |
|----|------|----------------------|----------------------------|
| P1 | hard-stop 计数器低于阈值清零 | `PROPOSED — NOT APPROVED` | `USER-APPROVED — IMPLEMENTED` |
| P2 | 同时越线时 hard-stop 优先 | `PROPOSED — NOT APPROVED` | `USER-APPROVED — IMPLEMENTED` |
| P3 | 状态写入失败即停训、不重试、不回退 checkpoint | `PROPOSED — NOT APPROVED` | `USER-APPROVED — IMPLEMENTED` |
| P4 | `--resume` 仅异常暂停；hard-stop 不可恢复 | `PROPOSED — NOT APPROVED` | `USER-APPROVED — IMPLEMENTED` |
| P5 | 暂停状态文件字段与异常策略 | `PROPOSED — NOT APPROVED`（10 字段版） | `USER-APPROVED — IMPLEMENTED`（**9 字段版**，见 §三） |
| P6 | 恢复前 config_checksum + 精确 step + vocab/seq 一致校验 | `PROPOSED — NOT APPROVED` | `USER-APPROVED — IMPLEMENTED` |
| P7 | hard-stop 安全收尾仅限日志 + 独立 JSON + 资源释放 | `PROPOSED — NOT APPROVED` | `USER-APPROVED — IMPLEMENTED` |
| P8 | 状态文件读取 ≠ 恢复授权 | `PROPOSED — NOT APPROVED` | `USER-APPROVED — IMPLEMENTED` |

**注**：本记录为事后归档；用户批准发生于 2026-10-26 的 B5 实施授权指令中，授权文本将 P1–P8 全部列为"必须遵守的既有项目规则"。

---

## 三、P5 字段数量冲突与裁决

### 冲突来源

- **R2.13**（`B5_Unresolved_Rules_Proposals.md`）P5 原文：10 字段（6 必要 + 4 可选），可选字段为 `config_checksum`、`model_checkpoint_path`、`validation_error_summary`、`timestamp`。
- **R2.16 用户指令**（2026-10-26）：明确"暂停状态 JSON 使用用户最新确认的 9 个字段：6 个必要字段 `version`、`step`、`warn_consec`、`hard_consec`、`pause_reason`、`exemption_window_passed`；3 个可选字段 `baseline_val_loss`、`config_checksum`、`timestamp`。不得自行增加第十个字段"。

### 裁决

以用户 2026-10-26 指令为准。9 字段版为**最终实现**。

| 字段 | 类型 | 必要/可选 | 说明 |
|------|------|----------|------|
| `version` | str | 必要 | 状态格式版本（`"1.0"`） |
| `step` | int | 必要 | 暂停时训练 step 号 |
| `warn_consec` | int | 必要 | 暂停时 warning 计数器值 |
| `hard_consec` | int | 必要 | 暂停时 hard-stop 计数器值 |
| `pause_reason` | str | 必要 | `nan` / `inf` / `failed` / `empty` / `format_error` / `hard_stop` |
| `exemption_window_passed` | bool | 必要 | 豁免窗口是否已通过 |
| `baseline_val_loss` | float \| null | 可选 | 基线 B |
| `config_checksum` | str \| null | 可选 | 模型配置 SHA-256 |
| `timestamp` | str | 可选 | UTC ISO-8601 |

**已移除**（相对 R2.13 原文）：`model_checkpoint_path`、`validation_error_summary`。
**新增**（相对 R2.13 原文）：`baseline_val_loss`。

**代码落实位置**：`aronalm/pause_state.py` L36-44（`REQUIRED_FIELDS` / `OPTIONAL_FIELDS` 元组，合计 9 字段，无第 10 字段）。

---

## 四、四种状态区分（当前 B5 各层）

| 层 | 状态 | 证据 |
|----|------|------|
| **用户决策已确认** | ✅ K1–K6（R2.9）；决定 A/B（R2.14）；P1–P8（R2.16 归档） | 各轮用户确认记录 |
| **代码已实施** | ✅ 提交 `d4fb934`（4 文件 +1251/−48）：`evaluator.py`、`pause_state.py`（新）、`trainer.py`、`tests/test_b5_early_stop.py`（新） | 仓库 diff、commit message |
| **测试已验证** | ✅ 75/75 全绿（50 旧回归 + 24 B5 专项 + 1 日志一致性修复测试），命令 `python -m pytest tests -q` | 本轮实际重跑记录 |
| **真实训练尚未验证** | ❌ `experiments/b5/` 未创建；B5 早停/异常暂停/恢复未经真实 500-step 训练验证；K1–K6 训练验收未达成 | 用户 2026-10-26 指令明确"真实训练实验尚未运行，不能声称 B5 已通过真实训练验收" |

**本记录不将前三项状态自动外推为第四项。**

---

## 五、实施中发现并修复的偏差

### 5.1 `trainer.py` 日志键一致性（已修复）

**问题**：`d4fb934` 中 `train()` 在 `EvalResult.status != "ok"` 时（旧调用方 + B5 关闭的异常观察场景）不再写入 `log_entry["val_loss"]` 与 `log_entry["val_perplexity"]` 键，导致 11 处旧调用方（`b4_4`、`b4_5` 脚本、`train_smoke.py` 等）读取 `val_metrics["loss"]` 时遇到 KeyError。

**修复**：`log_entry["val_loss"] = result.loss`（可能为 `None`）与 `log_entry["val_perplexity"] = result.perplexity`（可能为 `None`）**始终写入**；`val_status` 键仅在非 ok 时追加。

**影响**：旧行为**更接近**逐字节一致（键总是存在），新增 1 条测试 `test_legacy_trainer_log_shape_stable_on_anomaly` 覆盖此场景。

**未削弱任何安全约束**：仅改日志键存在性，未改计数器、未改 checkpoint 写入、未改恢复逻辑。

### 5.2 `pause_state.py` 未使用 import（已清理）

`from dataclasses import dataclass, asdict` 中 `asdict` 从未使用。已改为 `from dataclasses import dataclass`。无功能变化。

---

## 六、明确的安全不变量（重申）

以下约束在 `d4fb934` 代码中已落实，本记录再次强调**不得绕过**：

1. **Step 1–10 豁免窗口**：窗口内（含 Step 10）双计数器冻结；Step 11 起正常判断（K1/K2 + 决定 A）。
2. **双独立计数器**：`warn_consec` 与 `hard_consec` 独立维护，hard-stop 不依赖 warning（K3）。
3. **`>=` 语义**：阈值比较用 `>=`，精确等于阈值即计为一次（K3）。
4. **低于阈值清零**：warning 低于 warning 阈值清零（U2）；hard-stop 低于 hard-stop 阈值清零（P1，对称）。
5. **异常期间计数器保持原值**：NaN/Inf/失败/缺失/格式异常 → 暂停，不自动重试，不自动恢复（K4/K5/K6 + 决定 A.4）。
6. **hard-stop 优先**：同一次有效观察同时越线时 hard-stop 终止训练，warning 仅记录（P2）。
7. **hard-stop 后安全收尾**：仅写独立 9 字段 JSON + 日志 + 资源释放；**不写 checkpoint、不更新 best、不执行后续 step**（决定 B + P7）。
8. **状态写入失败**：停止训练，不重试，不回退到 checkpoint 或其他状态文件（P3）。
9. **恢复严格边界**：仅异常态可 `resume_from_pause`；须同时满足 状态文件有效 + config_checksum 匹配 + step 精确匹配（P4/P6）；**hard-stop 态永不可恢复**。
10. **读取 ≠ 授权**：状态文件存在、格式正确、校验通过均**不触发**自动恢复；仅显式 `--resume` 调用 `resume_from_pause()` 才可能恢复（P8）。
11. **旧 checkpoint 只读**：不自动迁移、不覆盖、不删除（U5）。

---

## 七、与历史文档的冲突登记

| 冲突 | 历史文档 | 当前实现 | 处理 |
|------|---------|---------|------|
| P5 字段数 | R2.13 P5 原文：10 字段（4 可选） | 9 字段（3 可选） | 以 2026-10-26 用户指令为准；本记录 §三 登记裁决；R2.13 原文**保留不改** |
| P1–P8 批准状态 | R2.14 标记 `NOT APPROVED` | 2026-10-26 用户批准并实施 | 本记录 §二 归档；R2.14 原文**保留不改** |
| `trainer.py` 日志键 | `d4fb934` 原实现仅在 ok 时写 `val_loss`/`val_perplexity` | 修复为始终写入（None 表示无效观察） | 本记录 §5.1 登记；后续提交将包含修复 |
| 真实训练验证 | 无 | 未执行 | 本记录 §四 标注"真实训练尚未验证"；不得声称 B5 已通过训练验收 |

**原则**：历史文档一律保留原文，本记录作为**可追溯增补**追加，不覆盖、不伪造过去的批准状态。

---

## 八、最终状态声明

```
DECISIONS A/B: USER-APPROVED — FINAL DECISION (R2.14)
K1-K6: USER-APPROVED (R2.9)
P1-P8: USER-APPROVED — IMPLEMENTED in d4fb934 (R2.16 归档, 2026-10-26)
P5 FIELDS: 9-field final form (6 required + 3 optional), supersedes R2.13 10-field proposal
CODE: d4fb934 + R2.16 后续修复提交 (log-key consistency)
TESTS: 75/75 passed (50 legacy regression + 25 B5 tests)
REAL TRAINING: NOT YET RUN — B5 早停 / 异常暂停 / 恢复 未经真实 500-step 训练验证
EXECUTION AUTH (L3-L8): NOT AUTHORIZED FOR NEW TRAINING
OVERALL: CODE COMPLETE, TESTS GREEN, PENDING REAL-TRAINING ACCEPTANCE
```

---

## 九、下一步（需用户单独确认）

1. **真实 B5 实验参数**：起点 checkpoint（推荐 `checkpoints_v2c_500steps/checkpoint_step350.pt`，B4-5 起点）、训练 steps（推荐 50–100）、eval 间隔（推荐 5–10）、本机资源预算（CPU、内存、预估耗时）。
2. **B5 实验脚本**：`experiments/b5/run_b5.py` 薄封装 `Trainer` + K 规则参数 + 决定 A/B + P1–P8 行为，跑真实 50-step 冒烟 + 100-step 主实验。
3. **训练验收报告**：对比 B4-5（无 K 规则）与 B5（K1–K6 生效）的触发点、计数器轨迹、异常恢复行为。

以上三项均属**高风险/长时任务**，需用户逐项确认后启动。

---

*R2.16 增补记录完成时间: 2026-10-26*
*类型: SUPPLEMENTAL DECISION RECORD — 不覆盖历史*
*状态: P1–P8 IMPLEMENTED · 真实训练未验证 · 需用户确认实验参数后启动 B5 实验*
