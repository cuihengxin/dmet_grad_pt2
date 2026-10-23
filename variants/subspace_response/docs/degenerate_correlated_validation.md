# bath简并的真实相关分子梯度验证

结论：在本次测试的两个小外部间隔LiF相关分子模型中，子空间response显著改善了解析梯度与独立能量有限差分的一致性。四重简并H2对照则新旧都正确，说明并非所有内部简并都需要产生精度差异。这比此前ROHF及合成谱测试提供了更直接的分子证据，但不是a1/NEVPT2生产体系验证。

## 模型与控制变量

所有结果均由实际分子电子积分、ROHF参考和嵌入空间全活性CASSCF（等价于该嵌入空间FCI）计算，不是人为设定本征值的矩阵模型。片段沿y方向相隔1000 Bohr，构造重复的bath谱；端点能量重新运行SCF、构建bath并求FCI，未使用解析梯度生成参考值。

- H2：四个H2片段，6-31g，每个片段第一个H的第一个AO为impurity；FCI(8电子,8轨道)。
- LiF：两个LiF片段，STO-3G，两个F的1s AO为impurity；FCI(8电子,6轨道)。这是为检查小外部间隔而选的核心轨道诊断分区，不是a1常用的活性区选择。
- AODMET默认Cholesky环境正交化，threshold=threshold2=1e-10。该AO分区实现的环境谱允许大于2，并按现有代码保留为bath；LiF两组二重简并合计4个值，正好满足原警告条件。H2则是一个四重组。
- SCF能量阈值1e-14、轨道梯度阈值1e-10，FCI阈值1e-14，所有成功样本检查收敛。
- legacy与subspace只切换response分母/内部旋转处理；能量、分区、求解器和阈值相同。两者统一使用GMRES以1e-10绝对阈值求外层Z，并检验真实残差，不使用Z初猜、历史解或方向回收。因此不是解方程速度比较。
- 数值梯度为第一个H或F沿z移动的中心能量差分，h=0.001及0.0005 Bohr；另检查x方向对称零分量。

## 主结果

下表是h=0.0005 Bohr的非零z分量绝对误差，单位Hartree/Bohr。

|分子模型|简并值数|最小bath/外部间隔|旧梯度误差|新梯度误差|
|---|---:|---:|---:|---:|
|4个H2，键长1.4 Bohr|4|1.177254e+00|5.682332e-08|5.682320e-08|
|2个LiF，键长5 Bohr|4|1.347484e-05|2.939244e-05|2.733485e-10|
|2个LiF，键长6 Bohr|4|4.754773e-06|7.452008e-05|3.937106e-11|

所有主结果新旧能量相同（本次记录中差为0），能量差分端点的bath数与冻结core数保持不变。Z方程真实缩放残差分别约为9.8e-12、3.4e-11、1.8e-14，排除了本次差异由Z求解精度造成的解释。

## 两个差分步长及实际梯度

|模型|h Bohr|数值梯度|旧解析梯度|新解析梯度|旧误差|新误差|
|---|---:|---:|---:|---:|---:|---:|
|4个H2，键长1.4 Bohr|0.001|-0.002603601265|-0.002603822869|-0.002603822869|2.216e-07|2.216e-07|
|4个H2，键长1.4 Bohr|0.0005|-0.002603766045|-0.002603822869|-0.002603822869|5.682e-08|5.682e-08|
|2个LiF，键长5 Bohr|0.001|-0.066404857634|-0.066375466065|-0.066404858774|2.939e-05|1.140e-09|
|2个LiF，键长5 Bohr|0.0005|-0.066404858501|-0.066375466065|-0.066404858774|2.939e-05|2.733e-10|
|2个LiF，键长6 Bohr|0.001|-0.052184532919|-0.052110013252|-0.052184533292|7.452e-05|3.727e-10|
|2个LiF，键长6 Bohr|0.0005|-0.052184533331|-0.052110013252|-0.052184533292|7.452e-05|3.937e-11|

LiF的新梯度与两种步长均一致，误差随步长减半下降；旧梯度误差则保持在约3e-5、7e-5的平台。H2新旧误差同样随步长减半约下降四倍，属于中心差分截断误差，未观察到新方法优势。不要把接近浮点噪声下限的单个误差比值当成普遍的“百万倍精度提升”。

## 阻尼参数的因果对照

在LiF 6 Bohr构型保持原legacy公式，只改变其自动阻尼参数，解析梯度逐步趋近新方法及数值梯度，确认主要误差来自全局阻尼，而非能量或波函数改变。

|legacy阻尼参数|旧解析梯度|与h=0.0005差分的误差|
|---|---:|---:|
|1e-10|-0.052110013252|7.452e-05|
|1e-12|-0.052180663080|3.870e-06|
|1e-14|-0.052184492895|4.044e-08|

新方法无需调小阻尼参数：内部bath旋转整体置零，外部响应在间隔非零时使用原始倒数。这些分子模型在外部间隔约1e-5量级时，新方法的适用机制已得到直接验证。精确外部简并、分区跳变或不收敛电子态仍不由这一处理解决。

## 未纳入成功表的尝试

- 两个LiF在8 Bohr时，收紧到上述精度的SCF未收敛；保存失败记录，不用于判定梯度方法的有效性。
- 四个LiF会形成12轨道的嵌入空间，超过脚本预设10轨道的计算保护上限，因此改用两个片段；这是控制FCI规模，并非response计算失败。
- 另保留LiF 3 Bohr的早期结果：新旧都接近数值梯度，精度优势较小。其收敛设置早于本表最终设置，不混入主结果。

## 回归与复现

完整独立包23项自动测试全部通过。新增test_correlated_degenerate_response.py独立重算LiF 6 Bohr，断言：4个简并值、稳定分区、两个差分步长均满足新误差小于2e-8、旧误差大于5e-5、新误差至少小1000倍，同时检查平移不变性。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python variants/subspace_response/tests/check_degenerate_correlated_gradients.py --case h2
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python variants/subspace_response/tests/check_degenerate_correlated_gradients.py --case lif_core --bond 5 --fragments 2
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python variants/subspace_response/tests/check_degenerate_correlated_gradients.py --case lif_core --bond 6 --fragments 2
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s variants/subspace_response/tests -p 'test_*.py' -v
```

逐点JSON、失败记录和测试脚本均在tests/degenerate_correlated_*及tests/check_degenerate_correlated_gradients.py。汇总位于tests/degenerate_correlated_summary.json。本轮只新增独立包中的测试及报告，没有将response合入主库。
