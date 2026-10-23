# Krylov 与跨几何初猜的实测比较

保持求解器为 PySCF Krylov，比较零初猜与映射上一步Z。独立包之外的原库源码不改动。

单线程6-31g，H4闭壳层/H5双重态，嵌入全活性空间CASSCF（FCI）。第二几何将第一个H沿z移动0.001 Bohr；每种配置5次，交替执行顺序，以下为中位数。计时包含完整梯度kernel，不包含外层SCF和DMET构建；Z-vector计时包括初猜对齐及残差检查。两组沿用同一能差预条件、subspace分母、1e-8求解阈值及收紧的Krylov线性相关阈值。

|体系|初猜|Krylov迭代|总响应作用|Z-vector ms|完整梯度 ms|真实缩放残差|
|---|---|---:|---:|---:|---:|---:|
|AODMET/H4|零|6|7|8.06|89.21|2.20e-08|
|AODMET/H4|映射上一步|4|7|8.55|85.98|9.54e-09|
|AODMET/H5|零|8|9|11.60|249.13|2.42e-08|
|AODMET/H5|映射上一步|5|8|11.61|245.85|2.56e-09|
|SSDMET/H4|零|6|7|7.94|81.38|2.01e-09|
|SSDMET/H4|映射上一步|4|7|8.99|83.86|1.88e-09|
|SSDMET/H5|零|8|9|12.98|255.86|2.01e-08|
|SSDMET/H5|映射上一步|4|7|10.13|247.42|1.51e-08|

总响应作用次数包含验证初猜、Krylov内部重算初猜残差和最终残差检查。有初猜的当前实现因此在内部迭代之外多两次响应作用；不能只按迭代数推断速度。

两配置梯度最大分量差 8.087e-10 Hartree/Bohr。中心差分步长0.0005 Bohr时最大误差 1.071e-07 Hartree/Bohr；将步长从0.001减半后误差下降。16项自动测试全部通过，新增Krylov初猜解与稠密线性方程解的比较及迭代减少检查。

SSDMET/H5 的Z-vector中位耗时从12.98降到10.13 ms，约降低22%；完整梯度从255.86降到247.42 ms，约降低3.3%。AODMET/H5的Z-vector耗时基本持平，两个H4实例则略慢。

结论：小体系中初猜减少内部迭代，但节省的矩阵作用少，对齐及验证开销可抵消收益；本次数据不能支持完整梯度有稳定加速，也不能预测a1大体系的加速比。

复现命令：
```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python variants/subspace_response/tests/benchmark_zvector.py --krylov-only --repeats 5
```

原始计时及梯度结果：tests/zvector_krylov_warm_results.json。初猜映射与共享缓存用法见README及zvector_gmres.md。
