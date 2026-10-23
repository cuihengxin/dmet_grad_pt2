# 固定J/K缓存与初始残差复用的计时

仅修改variants/subspace_response独立包；原库梯度源码保持原状。两项均为不改变响应方程的计算复用，新包默认开启，保留开关可分别对照。

## 实现

1. gen_contract_Z_A(mf, cache_static=True) 为每个几何点建立闭包。第一次响应调用计算单占据密度对应的J/K及其MO变换，同时保存固定kappa矩阵。后续调用直接复用；依赖当前Z的两自旋密度J/K仍每次重算。闭包只在当前几何有效，每次梯度都新建，不跨几何或SCF轨道更新复用。设False可恢复每次调用重算。采用惰性初始化，首轮固定计算成本也包含在Z-vector计时内。

2. 初猜检查已得到r0=b-Az0。启用残差复用时，Krylov/GMRES解 A delta=r0，再返回Z=z0+delta，避免求解器接收x0后重复计算Az0。历史组合初猜也保存最终验证的真实残差。无有效初猜时维持原流程。最终仍重新计算原始方程残差，容差及退出检查不变。

```python
# 创建grad后（新包默认均为True）
grad.zvector_cache_static = True
grad.zvector_reuse_residual = True
grad.zvector_solver = "krylov"
grad.zvector_warm_start = True
grad.zvector_cache = cache  # 共享上一步ZVectorCache
```

## 完整梯度基准

单线程、6-31g、H4闭壳层/H5双重态，AODMET与SSDMET均用全活性CASSCF（FCI）嵌入。Krylov、单步映射初猜、1e-8阈值，四种开关组合分别重复5次，轮换执行顺序。以下是相邻第二个几何点的中位数。Z-vector计时包含初猜、所有响应作用和最终残差检查；完整梯度为kernel总时间，不含外层SCF和DMET构建。

|体系|配置|Z-vector ms|完整梯度 ms|响应次数|固定J/K次数|
|---|---|---:|---:|---:|---:|
|AODMET/H4|基线|8.97|85.64|7|0|
|AODMET/H4|仅缓存J/K|8.67|83.25|7|0|
|AODMET/H4|仅复用残差|7.73|82.94|6|0|
|AODMET/H4|两项开启|7.91|82.32|6|0|
|AODMET/H5|基线|10.99|251.02|8|8|
|AODMET/H5|仅缓存J/K|10.46|254.30|8|1|
|AODMET/H5|仅复用残差|10.48|247.25|7|7|
|AODMET/H5|两项开启|9.47|257.28|7|1|
|SSDMET/H4|基线|8.90|85.28|7|0|
|SSDMET/H4|仅缓存J/K|8.78|86.38|7|0|
|SSDMET/H4|仅复用残差|7.81|87.97|6|0|
|SSDMET/H4|两项开启|8.11|91.26|6|0|
|SSDMET/H5|基线|11.97|255.17|8|8|
|SSDMET/H5|仅缓存J/K|10.82|270.39|8|1|
|SSDMET/H5|仅复用残差|10.93|261.52|7|7|
|SSDMET/H5|两项开启|9.64|268.22|7|1|

两项同时开启，小体系Z-vector中位耗时下降约9%～19%；完整梯度耗时受其余步骤及计时波动影响，本轮未显示稳定加速。闭壳层没有单占据J/K，缓存的收益主要只有固定小矩阵的构建，不能据此声称节省显著的J/K成本。

## 单独测试contract_Z_A

O2三重态、cc-pVDZ、28 AO，分别使用常规ROHF及density_fit ROHF。每次对20个固定随机响应向量求值，重复5批并交替顺序。计时包括创建响应闭包及第一轮缓存构建，不能把缓存初始化成本排除。

|O2响应计算方式|未缓存，20次 ms|缓存后，20次 ms|耗时降低|
|---|---:|---:|---:|
|常规|59.39|43.52|26.7%|
|density_fit|71.10|52.76|25.8%|

每批固定J/K调用从20次降至1次；依赖Z的响应J/K仍为20次。两种配置的响应矩阵结果逐元素一致。这是响应函数微基准，不是O2完整梯度或a1计时。

## 正确性与复现

20项自动测试全部通过。新增：实际闭壳层/开壳层响应缓存前后矩阵与J/K次数比较；Krylov及GMRES在单步/历史初猜下，复用残差均减少一次响应作用，并与独立稠密线性解一致。完整梯度四种开关配置在本次记录精度下完全一致；0.0005 Bohr中心能量差分最大误差 1.066e-07 Hartree/Bohr。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python variants/subspace_response/tests/benchmark_zvector_optimizations.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s variants/subspace_response/tests -p 'test_*.py' -v
```

原始逐次数据：tests/zvector_optimization_results.json。历史基准脚本benchmark_zvector.py默认显式关闭这两项以保留之前对照的可复现性；新建Gradients对象的默认值则均为True。尚未运行a1/Fe生产体系计时，不能外推固定加速比例。

两项默认开启后，NEVPT2单根、多根、六重态回归通过，5次内部相关响应求解全部收敛；单根细步长差分误差 4.211e-09、六重态差分误差 4.643e-09 Hartree/Bohr。结果见 tests/zvector_optimization_nevpt_regression.json。此回归不用于加速计时。
