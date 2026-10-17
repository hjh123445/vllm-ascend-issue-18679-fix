# vllm-ascend Issue #18679 修复 — 完整工作产物

修复 vllm-ascend 内存泄漏（ModelSlim MoE weight-loader 通过 reload metadata 泄漏整层权重）的全部分析材料。

## 文件导航（建议阅读顺序）

| 顺序 | 文件 | 说明 |
|---|---|---|
| 1 | `工作报告_vllm-ascend贡献.md` | 完整工作报告（中文），从 issue 检索到验证的全过程 |
| 2 | `repro_18679_real.py` | bug 复现脚本（40 行，入门第一步：跑它，看到 True True = 泄漏） |
| 3 | `modelslim_config_ORIG.py` | 修复前源码（看 79-106 行，出 bug 的闭包 wrapper） |
| 4 | `modelslim_config_PATCHED.py` | 修复后源码（看 82-160 行，绑定方法 wrapper） |
| 5 | `upstream_vllm_reload_analysis/sanitize.py` | ★上游 vLLM 的引用消毒机制（37 行，全链路最关键文件） |
| 6 | `upstream_vllm_reload_analysis/meta.py` | 上游快照拍摄（capture_layer_to_meta） |
| 7 | `upstream_vllm_reload_analysis/layerwise.py` | 上游 reload 编排（LAYERWISE_INFO 定义处） |
| 8 | `upstream_vllm_reload_analysis/routed_experts.py` | 上游 MoE 层的 weight_loader 定义 |
| 9 | `test_modelslim_moe_loader_leak.py` | 新增回归测试（5 用例，随 PR 提交） |
| 10 | `test_18679_leak.py` | 独立验证脚本（不依赖 vllm-ascend 安装即可跑） |
| 11 | `pr_18679_draft.md` | PR 描述（英文，含根因/修复/验证数据） |
| 12 | `pr_18679_files.patch` | 最终补丁（git apply 即可应用到 vllm-ascend） |

## 泄漏链一图流

```
LAYERWISE_INFO (value 强引用)
  → LayerReloadingInfo.restore_metadata
  → meta parameter.weight_loader   ← 闭包 wrapper（普通函数，逃过消毒）
      → 闭包 cell + __wrapped__
      → 原 bound loader (__self__)
      → layer → 整层真实权重 (NPU HBM)   ← 模型丢弃后仍被钉死 → OOM
```

修复：wrapper 改为 MethodType 绑定方法 → 上游 sanitize_layer_refs 的
`isinstance(value, MethodType) and value.__self__ is layer` 检查直接命中 →
绑定被正常剥离，泄漏链在第一环断开。

## 验证结果

- 复现脚本：orig 输出 `True True`（泄漏），patched 输出 `False False`（修复）
- 上游现有单测 70 passed + 新增回归测试 5 passed
- 910B 机器 quantization 全目录 487 passed（1 个 pre-existing 失败与补丁无关）

## 相关链接

- Issue: https://github.com/vllm-project/vllm-ascend/issues/18679
- 修复目标仓库: https://github.com/vllm-project/vllm-ascend
