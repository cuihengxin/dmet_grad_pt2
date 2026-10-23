# GMRES 与 Z-vector 初猜

仅修改 variants/subspace_response 副本，根目录梯度代码保持原状。AODMET、SSDMET 共用 embed_sim/grad/_zvector.py。

## 方程和停止条件

原程序求解 `(D + K) Z = -W`，其中 D 是独立轨道对能差，K 是原有 contract_Z_A 的反对称响应。新 GMRES 用 scipy.sparse.linalg.LinearOperator 实现 `A Z = Z + D^-1 K Z`，右端项 `b = -D^-1 W`。没有构造完整 Hessian，也没有给能差加位移或改变物理方程。默认 restart=80，最多200次内部迭代。callback_type='legacy' 确保 SciPy maxiter 统计内部迭代而非重启次数。绝对容差沿用 _krylov_conv_tol，rtol=0。

每次返回重新计算真实缩放残差 `A Z-b` 及原方程残差 `D(A Z-b)`，统计绝对范数和原方程相对残差。GMRES 非收敛会抛错；两种算法残差非有限或缩放残差超过 max(10*tol,1e-12) 都会抛错，不把不合格解交给梯度。10倍容差是返回后的浮点误差容限，GMRES自身仍按tol退出。Krylov原默认线性相关阈值可能使其在目标残差前退出；本副本将lindep收紧为 min(PySCF默认值,tol²)，保留原Krylov算法并尽量让两种方法达到相同精度，再检查真实残差。matvecs 包括初猜检查、求解和最终检查的所有昂贵响应作用；iterations 统计对应求解器的内部迭代；早期结果中的Krylov未记录次数，新增Krylov初猜基准已记录。

## 跨几何初猜

缓存保存前一步 mol、MO系数、占据数和Z；MO系数与Z都复制，避免后续原地修改污染缓存。缓存默认附着于 DMET 对象；跨新建对象需显式共享 ZVectorCache，见 README。

独立Z向量按 `(单占据,双占据)`、`(虚轨道,双占据)`、`(虚轨道,单占据)` 三块排列。先计算旧、新几何间的AO重叠 `S_old,new`，再得到 `O=C_old.T S_old,new C_new`。在双占据、单占据、虚轨道空间分别对 O 的对角块做 SVD，取极分解的正交因子 `U=L R.T`。旧解按 `Z_new[a,i]=U_a.T Z_old[a,i] U_i` 变换。该处理能吸收轨道的符号翻转、重排和同占据空间旋转，同时不混合不同占据类别；真实跨占据空间变化依然由新的响应方程求解。

原子/AO标签、基组、旋多重性、总电荷或MO占据不兼容时不用旧解；任一对齐块最小奇异值小于0.5时也放弃。映射后实际计算初猜残差，只有小于零初猜残差才使用。每个电子态需独立缓存，缓存不能自动识别所有电子态分支切换。仅支持实轨道，与当前梯度实现一致。

## 如何复现

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s variants/subspace_response/tests -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python variants/subspace_response/tests/benchmark_zvector.py
```

基准以6-31g的H4闭壳层和H5开壳层体系，分别测AODMET和SSDMET。主基准使用全活性空间 CASSCF（嵌入空间内等价于FCI），提供非零的相关响应。另保留ROHF检查于 tests/zvector_rohf_smoke_results.json；其右端项接近零，不用于判断加速。每种配置跑3次；计时仅包含完整grad.kernel，不含外层SCF和DMET构建。第二几何相对第一几何将第一个H沿z移动0.001 Bohr。对照配置为Krylov零初猜、GMRES零初猜、GMRES映射初猜。三者都用同一subspace分母方案，以隔离求解算法的影响；Krylov配置收紧了线性相关退出阈值并增加最终残差检查，因此它是精度对齐的原算法对照，并非原代码逐字计时；比未加诊断的代码多一次残差检查作用。所有配置阈值1e-8；记录冷启动与第二几何各自梯度、耗时及响应调用次数。独立比较梯度与中心能量有限差分，步长0.001和0.0005 Bohr。

原始计时样本保存于 tests/zvector_benchmark_results.json。小体系中重叠积分、SVD与额外残差检查可能抵消减少迭代的收益；不能直接外推Fe体系速度。此测试也不等同于a1/NEVPT2生产计算验证。

## 实测结果

单线程，每项三次的中位数；下面是第二几何点耗时，单位ms。H4为闭壳层，H5为双重态。

|方法/体系|求解器|完整梯度 ms|Z-vector ms|响应作用次数|GMRES内部迭代|真实缩放残差|
|---|---|---:|---:|---:|---:|---:|
|AODMET / H4|krylov|77.31|7.60|7|—|2.20e-08|
|AODMET / H4|gmres|79.93|9.42|9|7|9.26e-10|
|AODMET / H4|gmres + 初猜|79.01|8.80|8|4|9.49e-09|
|AODMET / H5|krylov|237.22|11.05|9|—|2.42e-08|
|AODMET / H5|gmres|244.21|14.64|11|9|3.16e-09|
|AODMET / H5|gmres + 初猜|234.96|11.99|9|5|2.54e-09|
|SSDMET / H4|krylov|74.90|7.21|7|—|2.01e-09|
|SSDMET / H4|gmres|83.83|9.08|8|6|2.01e-09|
|SSDMET / H4|gmres + 初猜|82.16|9.01|8|4|1.87e-09|
|SSDMET / H5|krylov|249.61|12.48|9|—|2.01e-08|
|SSDMET / H5|gmres|273.38|16.15|11|9|1.49e-09|
|SSDMET / H5|gmres + 初猜|246.36|12.41|9|5|1.82e-09|

三种配置梯度最大分量差为 8.127e-10 Hartree/Bohr；0.0005 Bohr中心差分的最大误差为 1.071e-07 Hartree/Bohr。各体系减半差分步长后误差下降，原始数值均保存在JSON。

初猜效果应同时看响应作用次数与耗时：对齐、验证初猜也有成本。小体系总时间约几十到几百毫秒，几个毫秒的差别会受缓存和运行波动影响，不据此声称生产体系有固定加速比。没有运行a1/Fe/NEVPT2的大体系对照。

## 回归验证

15项自动测试全部通过，包括四重bath简并的分子梯度、稠密线性解对照、轨道符号/同占据空间旋转、初猜拒绝与非收敛报错。

补跑默认GMRES的NEVPT2单根、多根和六重态回归，5次内部相关响应求解均收敛。单根中心差分最大误差 1.687e-08 Hartree/Bohr，六重态误差 4.643e-09 Hartree/Bohr。结果位于 tests/zvector_nevpt_regression.json。此项用于NEVPT2兼容性及梯度正确性验证，不作为初猜加速计时。
