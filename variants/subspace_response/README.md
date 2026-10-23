# 独立子空间响应版本

此目录保存当前项目的 embed_sim 库副本及本次新增的测试、结果和报告。项目根目录的原库已恢复修改前的梯度分母处理；原有用户修改保留。

使用新版本时，在导入任何 embed_sim 模块之前设置：

```python
import sys
sys.path.insert(0, "/lustre/home/2501110357/04grad/dmet_grad_pt2/variants/subspace_response")
from embed_sim import aodmet
from embed_sim.grad import aodmet as aodmet_grad
print(aodmet_grad.__file__)
```

此路径应位于其他 embed_sim 路径之前。请启动新的 Python 进程；已导入的模块不会随 sys.path 改变而自动切换。该副本使用本项目的 NEVPT2 实现，不会自动合并外部 nevpt_incore 副本的定制。

默认 `grad.embedding_response = "subspace"`；`"legacy"` 可对照原分母处理。

运行测试：

```bash
cd /lustre/home/2501110357/04grad/dmet_grad_pt2/variants/subspace_response
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_embedding_response.py -v
```

与原项目一致，该库是 namespace package，仍依赖环境中 LAMP_emb/embed_sim 的 BNO_bath 等共享模块；没有复制或修改这些外部依赖。

Z-vector 默认使用 GMRES，并在同一 DMET 对象上保存最近一次通过残差检查的解：

```python
from embed_sim.grad._zvector import ZVectorCache
z_cache = ZVectorCache()  # 放在优化回调外；每个电子态使用独立缓存
# 每个几何点创建 grad 后设置：
grad.zvector_solver = 'gmres'       # 'krylov' 对照原算法
grad.zvector_warm_start = True
grad.zvector_cache = z_cache       # 跨新建 DMET/Gradients 对象复用
grad.zvector_restart = 80
grad.zvector_max_cycle = 200        # 最大内部迭代次数
grad._krylov_conv_tol = 1e-7
# grad.kernel() 后读取 grad.zvector_stats
```

如果每步创建新的 DMET 对象，必须按上例显式共享缓存，否则缓存不会跨步保留。缓存仅存内存，不写 checkpoint。电子态改变时使用新的缓存。

实现与测试说明见 [Z-vector 报告](docs/zvector_gmres.md)。

Krylov 保持原求解算法并结合初猜的五次重复实测见 [Krylov 初猜报告](docs/zvector_krylov_warm.md)。设置 `grad.zvector_solver = "krylov"`，其余缓存用法相同。

最近多步最小残差组合初猜：设置 `grad.zvector_guess_mode = "history"` 并共享 `ZVectorCache(max_history=3)`。默认仍为 `"previous"`。实现、连续6个几何点实测及适用限制见 [多步初猜报告](docs/zvector_history.md)。

固定J/K缓存和初始残差复用默认开启：`grad.zvector_cache_static = True`、`grad.zvector_reuse_residual = True`，两者可分别关闭作对照。实现和实测数据见 [响应计算优化报告](docs/zvector_optimizations.md)。

实验性慢方向回收：`grad.zvector_recycle = True`，仅支持Krylov，默认关闭。本轮未发现总耗时加速，见 [回收实现与测试报告](docs/zvector_recycle.md)。

新增真实相关分子bath简并验证：两个LiF模型中，新response与中心能量差分一致，旧全局阻尼引入约3e-5至7e-5 Hartree/Bohr误差；H2四重简并对照新旧均正确。详见 [分子验证报告](docs/degenerate_correlated_validation.md)。
