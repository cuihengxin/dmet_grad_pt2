# 实验性慢方向回收：实现与未获得加速的测试结果

原库梯度源码保持不动，全部改动在独立包。zvector_recycle默认False。当前实验没有发现显著加速，不建议因此直接启用于a1生产优化。

## 回收的内容

普通初猜复用最终Z；本实验额外保留上一次Krylov搜索空间里响应较弱的少量方向。利用PySCF回调已经产生的向量及其响应，不增加提取阶段的响应调用。在归一化后的搜索空间中做SVD正交化，丢弃相对奇异值小于1e-10的方向，再从限制算符AQ中选择小奇异值对应的右方向。它们是搜索空间内的小响应方向，不是完整响应矩阵的精确本征向量，也不是harmonic Ritz回收；不保证对应所有慢收敛成分。提取的SVD时间计入总时间。

下一几何点，回收方向U与Z一样通过AO交叉重叠分占据块映射。重新计算当前AU并用SVD构造AB=C且C.T C=I；舍弃AU中相对奇异值小于1e-12的近零方向。先做粗空间修正，再用PySCF Krylov解投影后的修正方程：

```text
r = b - A z0
zbase = z0 + B C.T r
rperp = (I-C C.T) r
(I-C C.T) A delta = rperp
z = zbase + delta - B C.T A delta
```

该恢复公式保留原方程解；最后实际计算AZ-b验证，不用投影残差冒充原方程残差。当前实现只支持Krylov，开启recycle但solver=gmres会报错。

每次使用m个回收方向至少新增m次响应作用，以得到当前AU，恢复完整解还新增一次响应作用。第一几何没有历史空间可用，因此不能靠跨步回收加快首次求解，且首次提取还增加时间。默认只在搜索至少12步时提取最多8个方向，避免很短的求解也保存过多方向。

## 可选配置

```python
from embed_sim.grad._zvector import ZVectorCache
cache = ZVectorCache()  # 优化回调外创建，电子态改变时reset()
# 每次创建grad后：
grad.zvector_solver = "krylov"
grad.zvector_cache = cache
grad.zvector_warm_start = True
grad.zvector_recycle = True
grad.zvector_recycle_dim = 8
grad.zvector_recycle_min_iterations = 12
```

新增zvector_subspace_tol默认None，沿用_krylov_conv_tol。它只控制原Krylov内部向量范数/线性相关退出，不改变最终真实残差检查。若要收紧内部退出可显式设置，但这会增加迭代，并非回收加速。所有对照必须使用相同设置。

## 高迭代受控合成问题

147维、对称响应矩阵，8个较小谱值0.03～0.15，其余0.4～1.8，右端及矩阵在后两步小幅变化；不是分子梯度。实际方程容差1e-7，统一使用1e-16内部子空间退出阈值避免原求解器过早退出。3步序列重复3次，表中为后两步共6个样本中位数。所有结果均与独立numpy稠密解比较。

|配置|内部迭代|全部响应次数|总时间 ms|
|---|---:|---:|---:|
|零初猜|47|48|13.32|
|上一步Z|43|45|13.25|
|上一步Z＋回收8方向|36|47|15.89|

各配置最大相对解误差 8.740e-11。回收减少迭代，但总响应从45增加到47。另试4方向，迭代39、总响应46，也未减少总响应次数。毫秒级计时受运行波动影响。

更困难的先行合成尝试：将最小谱值降到0.002、右端尺度增大到1000，原Krylov在回收启动前就未通过真实残差检查；内部阈值1e-7时残差约1.34e3，收紧到1e-16后仍约3.79e-3。此失败情况不纳入成功计时表，也不能说明回收已经解决这种病态问题。

## 真实分子响应算符

O2三重态/cc-pVDZ，density_fit ROHF；键长2.3、3.2、4.0 Bohr，各连续3个几何点（每步+0.001 Bohr）。使用固定随机对称AO源经当前MO变换生成平滑右端，因此是真实分子响应算符，但不是NEVPT2能量梯度或实际a1右端。回收6方向。所有配置均开启固定J/K缓存和初始残差复用，统一1e-7实际容差及1e-16内部子空间阈值，每配置重复3次。以下为后两步6个样本中位数，时间包含回收向量映射、AU重算和SVD提取。

|O2键长 Bohr|配置|内部迭代|全部响应次数|总时间 ms|
|---|---|---:|---:|---:|
|2.3|零初猜|23|24|66.39|
|2.3|上一步Z|21|23|68.33|
|2.3|上一步Z＋回收6方向|20|29|95.41|
|3.2|零初猜|28|29|87.99|
|3.2|上一步Z|25|27|82.93|
|3.2|上一步Z＋回收6方向|23.5|32.5|107.88|
|4|零初猜|30|31|88.63|
|4|上一步Z|27|29|86.99|
|4|上一步Z＋回收6方向|26|35|108.87|

默认内部子空间阈值1e-7也另外测试并保留全部失败记录。2.3和3.2 Bohr的原Krylov解被真实残差检查拒绝，约为2.54e-5及1.6e-5；因此对应缓存没有合格历史，回收同样无法启动。4.0 Bohr通过检查，但单步初猜内部迭代中位15次、总响应17次；回收分别16.5和25.5，仍无收益。返回后检查容限沿用max(10*tol,1e-12)，不能把此处通过检查理解成残差严格小于tol。

## 正确性及结论

22项测试通过，包括独立稠密解、AB=C及投影恢复公式验证、多方向轨道映射/缓存重置，以及强制在小体系启用回收的AODMET和SSDMET真实梯度与不回收结果一致（绝对容差2e-8 Hartree/Bohr）。此前有限差分回归同时保持通过。

本轮回收只减少部分内部迭代，重新计算AU、投影及提取的成本抵消收益。默认保持关闭；没有测试a1/Fe大体系，也没有复现日志中133次的具体方程，不能声称可显著加速。日志前期r很大本身不是原方程真实残差，不能单凭它判断回收是否有用。首次几何没有历史可回收，应另从稳定的残差控制和预条件着手。

## 复现及数据

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python variants/subspace_response/tests/benchmark_zvector_recycle.py --dim 8
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python variants/subspace_response/tests/benchmark_zvector_recycle.py --dim 4
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python variants/subspace_response/tests/benchmark_zvector_recycle_molecular.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python variants/subspace_response/tests/benchmark_zvector_recycle_molecular.py --subspace-tol 1e-7
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s variants/subspace_response/tests -p 'test_*.py' -v
```

原始数据位于tests/zvector_recycle_synthetic_results.json（首次8方向结果）、zvector_recycle_synthetic_dim4_results.json、zvector_recycle_molecular_results.json、zvector_recycle_molecular_default_results.json。重新运行--dim 8会写入带dim8名称的新结果。
