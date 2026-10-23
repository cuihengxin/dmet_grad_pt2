# 多步最小残差 Z-vector 初猜

实现仅在独立包中，原库梯度源码不变。保持 Krylov，比较零初猜、上一步解和最近3步解的最小残差组合。默认仍为 previous；history 需显式开启。

## 实现

缓存分别保存最近3个收敛解及各自的几何、MO系数、占据数。每个历史解通过旧/新AO交叉重叠，在双占据、单占据、虚轨道空间内分别极分解对齐，得到当前轨道规范中的列向量V。对当前缩放方程 AZ=b，逐列计算Y=AV；按Y各列范数归一化后，用SVD最小二乘求 min ||Yc-b||，rcond=1e-8，丢弃近线性相关方向，再还原权重。初猜为Vc。

组合候选会重新做一次当前响应作用以检查真实残差，只有优于单步及零初猜才接受；否则回退。不改变响应矩阵、预条件或最终求解方程。占据/基组/体系不兼容会清空缓存；轨道空间变化过大时跳过对应历史。电子态改变需要调用cache.reset()或更换缓存。

这不是完整Krylov子空间回收，只复用几个历史解。3步历史需要3次当前响应作用，加上组合验证1次；Krylov的x0处理还会再次计算初始残差，因此减少内部迭代不必然减少总响应作用。计时和matvecs均包含这些成本。

## 使用

```python
from embed_sim.grad._zvector import ZVectorCache
cache = ZVectorCache(max_history=3)  # 几何优化回调之外；也可设为5

# 每个几何点创建grad后：
grad.zvector_solver = "krylov"
grad.zvector_warm_start = True
grad.zvector_guess_mode = "history"
grad.zvector_cache = cache
```

单步对照使用guess_mode="previous"；零初猜使用warm_start=False。配置同样支持GMRES，本次基准只用Krylov。

## 测试设置

单线程6-31g；H4闭壳层、H5双重态；嵌入空间全活性CASSCF（FCI），求解容差1e-8，各配置均使用收紧的Krylov线性相关退出阈值。连续6个几何点沿小幅弯曲路径变化，每种配置完整序列重复3次、轮换配置顺序。完整梯度计时不含外层SCF和DMET构建。下表只统计第4～6个点，缓存已积累3步历史；数值是3次重复×3个几何点的9个样本中位数，并非同一几何点的9次计时。

|体系|初猜|内部迭代|总响应作用|Z-vector ms|完整梯度 ms|初始缩放残差|
|---|---|---:|---:|---:|---:|---:|
|AODMET/H4|zero|6|7|8.43|82.73|2.99e-01|
|AODMET/H4|previous|4|7|8.49|81.22|6.73e-04|
|AODMET/H4|history|2|8|11.12|87.93|6.69e-07|
|AODMET/H5|zero|8|9|11.72|253.07|3.11e-01|
|AODMET/H5|previous|5|8|11.45|263.71|2.35e-04|
|AODMET/H5|history|2|8|13.44|251.75|3.37e-07|
|SSDMET/H4|zero|6|7|8.47|85.48|3.52e-02|
|SSDMET/H4|previous|4|7|8.75|84.44|1.03e-04|
|SSDMET/H4|history|1|7|9.64|84.49|3.05e-08|
|SSDMET/H5|zero|8|9|12.04|254.22|7.56e-02|
|SSDMET/H5|previous|5|8|11.85|257.45|1.05e-04|
|SSDMET/H5|history|2|8|13.02|252.00|7.11e-08|

## 结论与正确性

多步组合将初始残差再降约几个数量级，内部迭代明显减少；当前小体系中，组合初猜的额外响应、交叉重叠与SVD开销抵消了收益，Z-vector总时间比单步初猜略慢。完整梯度计时有波动，不能据此宣称加速。保留history可选，默认previous不变。尚未测试a1大体系。

所有几何点/配置的梯度最大分量差 1.244e-09 Hartree/Bohr；最后几何点中心差分步长0.0005 Bohr的最大误差 1.057e-07 Hartree/Bohr，步长从0.001减半后误差下降。18项自动测试通过，新增独立稠密线性解检验、多历史组合精确张成目标解、秩亏重复历史及缓存重置测试。

复现：
```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python variants/subspace_response/tests/benchmark_zvector_history.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s variants/subspace_response/tests -p 'test_*.py' -v
```

原始逐点数据：tests/zvector_history_results.json。
