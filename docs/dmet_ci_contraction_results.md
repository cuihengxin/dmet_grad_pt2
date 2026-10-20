# DMET–FIC-NEVPT2 CI 源收缩移植与验证

DMET 梯度默认使用 `contracted`，通过 `solver_option['ci_response_backend'] = 'legacy'` 可切回旧实现。只替换 RDM 权重对 CI 的导数构造，继续复用原有单电子响应、半正则化响应、投影、内外层 Z-vector 和核梯度组装。收缩核心直接复用 `embed_sim.fic_nevpt.grad.ci_contraction`，保留内存检查和统计。

## 测试方法

- Python 3.11，PySCF 2.14.0，OpenMP/BLAS 单线程。
- 非等距 H 链，STO-3G，坐标 z_i = 1.1 i + 0.15 (i mod 2) Å，前若干 AO 为 impurity。嵌入空间确实小于完整体系。
- 两个 singlet 的等权 SA-CASSCF，目标 state=0；CAS(4,4) 使用 Newton，CAS(6,6) 使用标准 CASSCF，均要求参考态收敛。
- 新旧后端使用完全相同的收敛参考态，分别创建新的 NEVPT/梯度对象；检查内部 Z-vector 收敛。
- 先各预热一次，再交替顺序测量各三次，报告中位数。保留完整核梯度、CI 源项、响应 RHS、Z-vector、relaxed 1/2-RDM 和 X 原始数组。
- “完整梯度耗时”包含 NEVPT 初始化、能量、内部 relaxed 密度响应和外层 DMET 核梯度；不包含共享 SCF、嵌入构建和 CASSCF。不能直接解释成从头开始所有计算的加速比例。

## 主要结果（canonstep=1）

| 活性空间 | 全体系原子 / impurity AO / 嵌入轨道 | CI 维数 | 最大核梯度差 Eh/Bohr | CI 源项旧/新 s | 源项加速 | 完整梯度旧/新 s | 完整梯度加速 |
|---|---|---|---|---|---|---|---|
| CAS(4,4) | 8 / 3 / 6 | 36 | 6.665e-12 | 0.267103 / 0.006376 | 41.89× | 0.721467 / 0.458404 | 1.57× |
| CAS(6,6) | 10 / 4 / 8 | 400 | 9.494e-11 | 22.471756 / 0.066601 | 337.41× | 24.348922 / 1.950585 | 12.48× |

两个算例能量差均为零。CAS(6,6) 最大 CI 源项差约 1.78e-11，最大 relaxed 2-RDM 差约 1.34e-9；投影后 RHS、Z-vector 和最终核梯度差异见原始 JSON。测试断言所有对比量的最大绝对差均小于 1e-8，不要求浮点逐位一致。

## 补充覆盖

- `tests/test_ci_response.py`：常规/DF/DMET 三类梯度；不同自旋、1–4 阶非对称随机权重、低电子数；新路径不调用 transition RDM；无效后端和内存不足检查。四个 unittest 方法全部通过。
- CAS(4,4) 的 canonstep=0 和 2 完整 DMET 核梯度对比分别为 6.67e-12 和 2.91e-12 Eh/Bohr；各一次正式配对，主要用于数值回归。
- 未做大型真实分子、DMET 的 DF 积分端到端基准或有限差分核梯度验证。本测试验证相对旧实现的等价性，绝对梯度精度仍沿用原实现。

## 运行依赖与兼容适配

当前仓库缺少 cahf、BNO_bath、sacasscf_mixer 等完整 DMET 支持模块。基准脚本将本机 `/home/cuihx/Desktop/17grad/DMET-grad/embed_sim` 追加到 namespace 搜索路径，仅补齐缺失模块；当前仓库的 ssdmet、grad.solvers、dmet_nevpt 均优先使用，实际模块路径记录在 JSON。

外层现有 `get_grad_Z_vector` 要求 spin-resolved `make_rdm1()`，不能直接处理 RHF 的二维密度。脚本在完成 RHF 嵌入构建后，把完全相同的已收敛闭壳层轨道/能量存入 ROHF 对象供外层响应使用；此适配同等用于新旧后端，未修改生产代码的外层算法。构造参考态时用到的 mock 只阻止重复 CASSCF；两个后端的 NEVPT 能量和完整核梯度均实际执行。

## 复现命令

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/home/cuihx/miniconda3/envs/psi4env/bin/python tests/benchmark_dmet_ci_response.py \
  --dependency-path /home/cuihx/Desktop/17grad/DMET-grad/embed_sim \
  --atoms 8 --impurity 3 --ncas 4 --optimizer newton --output tests/dmet_ci_cas4.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/home/cuihx/miniconda3/envs/psi4env/bin/python tests/benchmark_dmet_ci_response.py \
  --dependency-path /home/cuihx/Desktop/17grad/DMET-grad/embed_sim \
  --atoms 10 --impurity 4 --ncas 6 --optimizer standard --output tests/dmet_ci_cas6.json

PYTHONDONTWRITEBYTECODE=1 /home/cuihx/miniconda3/envs/psi4env/bin/python -m unittest discover -s tests -v
```

补充 canonstep 测试在 CAS(4,4) 命令后添加 `--canonstep 0` 或 `--canonstep 2 --repeats 1`，另指定输出文件。
