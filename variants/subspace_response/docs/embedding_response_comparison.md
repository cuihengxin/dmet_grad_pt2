# bath 简并响应：实现与实测

## 警告的含义

原来的 `4 degenerate eigenvalues ...` 表示环境密度矩阵中有四个被归入简并组的 bath 占据数（判据 `1e-8`）；不意味着全部四个一定属于同一个简并组，也不表示能量本身发散。旧代码据此自动将所有 bath 和 core 分母替换成 `gap/(gap**2+1e-10)`。

单个简并本征向量的导数不唯一，但完整保留的、与外部隔开的 bath 子空间的投影算符可以有唯一导数。其一阶变化仅含 bath 与外部轨道的混合；bath 内部的反对称旋转在投影导数中抵消。因此屏蔽整个 bath–bath 块有数学依据，条件是下游能量/求解器对该内部旋转不变且所选解分支光滑。不能单凭消除警告判断完整梯度正确。

## 代码和使用

`embed_sim/grad/_embedding_response.py` 提供公共响应函数，两种 DMET 的 `get_grad_I` 均已接入；后续收缩、内部 CASPT2/NEVPT2 响应方程未改动。两种梯度类默认使用 `subspace`。

```python
grad = dmet.Gradients()
grad.embedding_response = 'subspace'  # 新方法，默认
grad._perturb_anyway = False         # 不因内部简并自动加阻尼
# 其他 solver_option 与原来相同
new_gradient, energy = grad.kernel()

grad.embedding_response = 'legacy'   # 原方法，含原来的自动阻尼和日志
old_gradient, old_energy = grad.kernel()
```

原始两份分母分支逐行保存在 `_embedding_response_legacy.py` 中，通过运行时开关调用。未复制整个梯度程序，避免两份收缩代码逐渐产生差异。原有用户修改的 `embed_sim/aodmet.py` 未改动。

相对于题目给出的函数，增加了以下边界处理：

- 外部或实际需要的 core 分母精确为零时，无阻尼模式抛异常，避免把不存在唯一解的问题默默置零。
- 显式启用 `_perturb_anyway=True` 时允许阻尼并发出近似梯度提示；bath–bath 块仍为零。
- bath–外部或 core–补空间间隙小于 `1e-8` 时提示风险，不自动阻尼。
- 无 frozen core 时返回零的未使用 core 因子，避免无物理贡献的分母触发异常。

输入必须是 `[冻结虚轨道 | bath | 冻结占据轨道]`。当前 AODMET 已显式重排；SSDMET 使用排序后的本征值和连续阈值区间。保留原来的 `2 - occupation` core 近似，未改成逐个实际 core 本征值的差。

## 验证结果

运行环境：Python 3.12.7、NumPy 1.26.4、PySCF 2.13.0，单线程。原始数据保存在 `tests/embedding_response_results.json`。

1. 人造八维谱，四个 bath 占据数均为 0.4，外部为 0/2。投影算符中心差分最大误差随步长 `1e-4 → 5e-5 → 1e-5` 从 `9.12e-7 → 2.28e-7 → 9.13e-9` 下降。随机旋转四重简并 bath 后，投影导数最大变化 `6.66e-16`。这验证了响应层的导数和规范不变性。
2. 四个相隔 1000 Bohr 的相同 H2，6-31g，前四个 H 的 AO 为杂质；这是刻意构造的四重 bath 简并检查。AODMET 的 bath 占据数约为四个 0.595656，SSDMET 约为四个 1。ROHF 求解器下，两种 DMET 新旧能量和梯度相同到所记录精度。步长 `5e-4` Bohr 时，所检查梯度分量的有限差分误差约 `5.50e-8 Hartree/Bohr`；`1e-3` 时约 `2.20e-7`。**该良态例没有显示新方法更准确，说明内部简并不必然导致坏梯度。**
3. 边界附近的人造谱 `[0, 1e-6, 1.001e-6, 0.7, 2]` 中，内部近简并触发旧方法的全局阻尼。一个真实 bath–外部分母的精确倒数为 `1e6`，旧方法返回约 `9900.99`，新方法保持 `1e6`。这显示旧方法可压低真实响应约 101 倍；同时如此大的真实响应本身说明问题病态，不能把“大梯度”视为数值质量保证。这个例子比较响应因子，不是实际分子的梯度精度证明。
4. 已有 `check_single_root_gradient.py` 在新默认分支下全部通过，包括 NEVPT2、内部 CI legacy/contracted 对照、单根/多根与 sextet 检查。单根有限差分误差 `4.21e-9`，sextet 为 `4.64e-9 Hartree/Bohr`。输出另存 `embedding_response_nevpt_regression.json`。这里的内部 CI legacy 对照与本次 embedding legacy 开关是不同的对照。

另保留一个失败诊断：方形 H4/6-31g、对角两个 H 为杂质，ROHF 计算中新旧梯度都偏离有限差分约 `0.061 Hartree/Bohr`。该点未得到目标 bath 简并；不能用它评价简并修复成功，也不能把误差归因于这次修改。平方体系的 SCF 分支、阈值附近 bath 数目、嵌入解连续性等需另行诊断。数据在 `embedding_response_square_diagnostic.json`，可用 `--square-diagnostic` 重跑；该模式只报告，不断言完整梯度正确。

## 效率和适用边界

两种方法均分配 `nbath × nenv` 分母，主要时间和内存复杂度相同。新方法去掉简并分组，增加掩码和有效性检查。500 个环境轨道/100 个 bath 的直接因子函数计时约为每次 0.14 ms（新）和 0.19 ms（旧），具体重复测量以 JSON 为准。此微基准不包含完整 DMET；ROHF 分子 kernel 单次约 0.05 秒，新分支首次调用还受导入/预热影响，不能据此声称整体提速。大体系主要成本仍是积分、嵌入求解和 Z-vector。

以下情况仍不能靠本方案获得可靠唯一梯度：bath 与舍弃空间的谱隙关闭；阈值穿越导致 bath 维数变化；冻结 core/补空间近简并；CAS 活性空间、目标态或 SCF 解发生跳变；求解器不收敛或能量不具有所需的 bath 旋转不变性。需要检查实际体系的谱隙，并对多个步长、多个方向进行能量有限差分。尚未运行用户实际出现警告的大分子，也未对 CASPT2 的完整梯度重新验证。

## 复现

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/check_embedding_response.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/check_embedding_response.py --square-diagnostic
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/check_single_root_gradient.py --output /tmp/embedding_response_nevpt.json
```

## 自动正确性测试（2026-10-23 补充）

新增 `tests/test_embedding_response.py`，共 11 项 unittest，实际运行全部通过。测试以重新对角化得到的投影或能量中心差分为独立参考，覆盖：精确四重简并、近简并、非简并的投影能量导数；冻结 core 投影导数；投影导数切空间条件；简并 bath 随机旋转不变性；小外部谱隙下新旧响应的误差；外部精确简并的异常及显式阻尼；空子空间和非法输入；两种 DMET 分支切换和警告；真实四重简并分子的梯度。

小外部谱隙的人造谱测试要求新响应的投影导数相对有限差分误差小于 `1e-4`，同时确认旧自动阻尼的误差大于 `10%`。这是投影响应的验证，不代表实际大分子的梯度改善幅度。

分子检查增加一个由不同原子坐标组成的混合位移方向，两个方向均使用 `1e-3`、`5e-4` Bohr 步长；断言最终误差小于 `1e-6 Hartree/Bohr`，且步长减半后误差至少下降约一半（另有舍入容差）。同时断言新旧能量一致、梯度有限、总平移梯度接近零。原来的方形 H4 失败诊断仍保留，不列为正确性通过例。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_embedding_response.py -v
```

这些测试确认所列情形中的实现正确性；未覆盖用户实际大分子、CASPT2 完整梯度、bath 维数跳变或所有能量求解分支。
