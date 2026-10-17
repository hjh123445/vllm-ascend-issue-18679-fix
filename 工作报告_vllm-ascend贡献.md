# vLLM-Ascend 开源贡献工作报告

**日期**: 2026-10-17
**目标**: 为 vllm-ascend / sglang(Ascend) 社区贡献有技术深度的修复，产出可提交的 PR
**工作目录**: `/data/ifly_jyhe23/hot_model_top5/audio_understand/qwen-audio/oss_contrib_20261017/`（全部文件隔离于此）

---

## 一、Issue 检索与筛选

### 检索范围
- `vllm-project/vllm-ascend`：691 个 open bug（label=bug）
- `sgl-project/sglang`：65 个 Ascend 相关 open issue/PR

### 筛选标准
1. 有技术深度（涉及内存管理/引用语义/分布式正确性等，能体现工程能力）
2. 不太热门（无已认领 PR、回复少，竞争小）
3. 本机 910B 可复现验证（或纯 CPU 可验证）

### 候选评估结论

| Issue | 标题 | 状态 | 结论 |
|---|---|---|---|
| #18209 | 细粒度 TP rank layout 错误致静默挂起 | 已被 PR #18373 修复关闭 | 放弃 |
| #17932 | NPU KV-offload 拷贝线程静默死亡 | 已有 open PR #17933 认领 | 放弃 |
| **#18679** | **ModelSlim MoE weight-loader wrapper 通过 reload metadata 泄漏整层权重** | 无人认领，0 回复，10-14 新报 | **✅ 选定** |
| #17797 | chunked prefill 下 embedding 输出错乱（310P） | 无人认领，但需 310P 硬件 | 备选 |
| #17874 | GLM-4.6V vision encoder 910C 无有效 TP | 需要 910C/A3 环境 | 备选 |

**选定 #18679 的理由**：
- 根因是 Python 引用语义层面的内存泄漏（弱引用字典 + 闭包 + `functools.wraps` 的 `__wrapped__` 链），技术深度高、面试可讲；
- 纯 CPU 即可复现验证（issue 自带 minimal reproduction），不依赖特定 NPU 型号；
- 影响真实生产场景（ModelSlim 量化 + MoE + RFork 模型替换时 OOM），修复有实际价值；
- 涉及 vLLM 上游 reload 机制（`LAYERWISE_INFO` / `sanitize_layer_refs` / `capture_layer_to_meta`）与 vllm-ascend 量化代码的交互，跨两层代码库。

---

## 二、根因分析（Issue #18679）

### 泄漏链条

`vllm_ascend/quantization/configs/modelslim_config.py` 中 `_make_modelslim_moe_weight_loader` 用**闭包普通函数**包装 MoE 层的 bound weight loader：

```python
@wraps(weight_loader)          # ← 复制了 __wrapped__ 指向原 bound method
def modelslim_moe_weight_loader(...):
    return weight_loader(...)  # ← 闭包 cell 也持有原 bound method
```

模型初始化时，vLLM 上游的 layerwise reload 机制调用 `record_metadata_for_reloading` → `capture_layer_to_meta`，把参数属性（含该 wrapper）拷贝进 meta tensor，存入全局 `LAYERWISE_INFO: WeakKeyDictionary`。

上游的防护 `sanitize_layer_refs` 只处理**直接**的 `MethodType` 属性（`value.__self__ is layer` 时剥离绑定）。但 wrapper 是普通函数：
- 闭包 cell 里的 bound method 不在 `tensor.__dict__` 顶层；
- `functools.wraps` 复制的 `__wrapped__` 属性同样指向 bound method；

于是形成引用链：

```
LAYERWISE_INFO (values 强持有)
  → LayerReloadingInfo.restore_metadata
  → meta parameter.weight_loader (wrapper)
  → closure cell + __wrapped__ → 原 bound loader
  → __self__ → layer → 真实权重 (NPU HBM)
```

WeakKeyDictionary 的 key 是弱引用，但 **value 强持有**，meta 参数值间接引用 layer → 模型丢弃后 layer 和全部 MoE 权重永不释放。在 64GB NPU 多节点部署 DeepSeek V4.1 Flash（ModelSlim 量化 + 专家并行）触发 RFork 回退时加剧 OOM。

### 验证（复现）

issue 中的 minimal repro（CPU 即可跑）在未修复代码上输出 `True True True`（layer/weight 均被保留），与 issue 报告一致。

---

## 三、修复方案

### 核心思路：让 wrapper 成为上游 sanitize 协议的"一等公民"

把闭包 wrapper 改为**模块级函数 + `MethodType` 绑定到 layer**：

1. `_bound_modelslim_moe_weight_loader(layer, ...)` — 模块级实现（不反向引用任何 layer）；
2. 原 loader 的**未绑定函数**存到 layer 实例属性 `_modelslim_orig_weight_loader`（函数对象不引用 layer，无泄漏）；
3. `_make_modelslim_moe_weight_loader` 返回 `MethodType(_bound_modelslim_moe_weight_loader, layer)`。

效果：
- wrapper 本身是 `MethodType` 且 `__self__ is layer` → `sanitize_layer_refs` 正常剥离、`restore_layer_refs` 正常恢复（reload 往返兼容）；
- 无闭包 cell、无 `__wrapped__` → 无隐藏引用；
- 幂等保护：重复包装时直接返回现有 wrapper（`__func__ is` 判断），避免无限递归；
- 兼容现有单测：非 bound loader（MagicMock）回落到闭包形式（mock 无 `__self__`，不存在泄漏面）。

### 行为保持
名字映射（`w2_scale_bias → w2_scale` 等）、参数透传、返回值语义完全不变。

---

## 四、验证结果汇总

| 验证项 | 环境 | 结果 |
|---|---|---|
| 泄漏复现（原始代码） | 本机 CPU + 真实 vllm reload 机制 | `True True` — 复现 ✅ |
| 泄漏修复（打补丁） | 同上 | `False False` — 修复 ✅ |
| 行为（映射/委托/返回值/幂等） | 本机 | 全部保持 ✅ |
| 上游现有 modelslim 单测（70 个） | 本机 py311 | 70 passed + 15 subtests ✅ |
| 新增回归测试（5 个） | 本机 py311 | 5 passed ✅ |
| **quantization configs 全目录** | **远端 910B (172.28.30.17)** | **115 passed ✅** |
| **quantization 全目录（4 类配置）** | **远端 910B** | **487 passed**（1 个 pre-existing 失败与补丁无关，未打补丁同样失败）✅ |

环境说明：
- 本机：910B3×8（卡 1-7 空闲使用）、CANN 8.5.0、conda env `va311`（Python 3.11 + torch 2.10.0 + torch-npu 2.10.0.post4 + vllm 0.30.0）
- 远端：172.28.30.17（910B3×8 全空闲）、CANN 8.2.RC2、独立 env `va311`（同上）
- 全部安装在独立目录/独立 conda env，未触碰机器上已有环境
- 注：vllm-ascend main 分支依赖 vllm main（超前于 release），单测验证基于与 vllm 0.30.0 wheel 匹配的 `v0.30.0rc2` tag（wrapper 代码与 main 完全一致，已 diff 确认）

---

## 五、产出物清单

| 文件 | 说明 |
|---|---|
| `pr_18679_files.patch` | **最终 PR 补丁**（modelslim_config.py 修改 + 新增测试文件，git diff 格式） |
| `test_modelslim_moe_loader_leak.py` | PR 附带回归测试（5 个用例） |
| `pr_18679_draft.md` | PR 描述草稿 |
| `modelslim_config_orig.py` / `modelslim_config_patched.py` | 修改前/后完整文件（基于 main 分支） |
| `repro_18679_real.py` | issue minimal repro（真实模块版） |
| `test_18679_leak.py` | 独立泄漏验证脚本（不依赖 vllm-ascend 安装） |
| `va_repo/` | vllm-ascend main 分支源码（逐文件经 GitHub raw 拉取） |
| `va_repo/va_rc2/` | v0.30.0rc2 tag 源码（已打补丁，单测验证基座） |
| `issues/` | issue 检索缓存 |

---

## 六、提交流程（待用户执行）

1. fork `vllm-project/vllm-ascend` 到自己账号
2. `git checkout -b fix/modelslim-moe-loader-leak`
3. `git apply pr_18679_files.patch`（已在干净 rc2 基座上验证 `git apply --check` 通过，本地与远端 910B 双重演练成功）
4. 本地跑 `pytest tests/ut/quantization/configs/ -q`
5. push 后开 PR，标题 `[BugFix][Quantization] Break the layer-reference chain in the ModelSlim MoE weight loader wrapper`，正文用 `pr_18679_draft.md`，`Fixes #18679`

> 提交前建议在 issue #18679 下留言说明正在处理（"I'm working on a fix"），避免与他人撞车。

---

## 七、后续可做（本次未完成）

- **#17797**（chunked prefill embedding 错乱）：需要 310P 硬件，本机为 910B 无法直接复现；可先做代码级分析（chunk 边界 + pooling last-token 索引）在 issue 下讨论。
- 本修复的 e2e 验证（真实 ModelSlim 量化 MoE 模型加载→丢弃→显存回收曲线）：需要公开的 ModelSlim 量化 checkpoint（如 ModelScope 上的 w8a8 量化 DeepSeek/Qwen MoE 模型），下载量大，未在本轮完成。
- sglang 社区的 NPU issue（如 #40296 min_p 采样 TypeError、#40516 FuseEP 共享专家路由）作为第二贡献点。

---

## 八、环境/网络问题备忘（对后续工作有用）

- 本机 git clone 走代理握手失败，但 `curl`（raw.githubusercontent.com / api.github.com / codeload）可用——源码获取全部走 HTTP API + raw
- PyPI 单连接仅 ~20KB/s，写了 `pdl.sh`（16 路分段并行 + 断点续传，~600KB/s）下载大 wheel
- `vllm-ascend` main 与 vllm release wheel 存在 API 漂移（`prepare_dcp_local_seq_lens` 等），单测验证需配对 tag（rc2 ↔ vllm v0.30.0）
- 远端 172.28.30.17 SSH 免密可用，`va311` conda 环境已就绪，测试基座 `~/oss_contrib_20261017/va_rc2` 已同步
