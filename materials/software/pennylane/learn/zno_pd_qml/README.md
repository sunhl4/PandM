## ZnO@Pd 包覆过程：从 LAMMPS dump 到 QML（Quantum Kernel）建模

你已经有：
- LAMMPS/ReaxFF 轨迹（dump 坐标）
- 温度 \(T\)、Pd 团簇直径 \(d_{\mathrm{Pd}}\)、表面 O 浓度/缺陷（9个模型）
- 已定义的包覆指标（包覆原子数 / Pd 表面积），以及 MSD/扩散系数等

这里提供一个可执行的 pipeline：
1. 读 LAMMPS `dump custom`（无需键级）
2. 计算包覆程度/包覆速率 + Pd/基底重构 + Zn/O 协同攀爬等 CV（collective variables）
3. 按“时间窗”把轨迹切成训练样本（避免把整条轨迹当一个点，信息太少）
4. 用 **Ridge / RBF-KRR / Quantum-KRR**（以及可选 SVC）做回归/分类
5. 支持 **按温度/尺寸/O模型外推** 的 group split（更符合你要写的“规律/机制”）

---

## 1) 你需要准备什么

### 1.1 dump 格式要求（推荐）
推荐 LAMMPS 输出（示例）：

```lammps
dump 1 all custom 100 dump.lammpstrj id type x y z
dump_modify 1 sort id
```

也支持 `xs ys zs`（会用 box bounds 自动换算到真实坐标）。

### 1.2 原子 type 映射
你需要提供 “type -> 元素” 的映射，例如：

```json
{
  "Zn": [1],
  "O":  [2],
  "Pd": [3]
}
```

---

## 2) 配置文件：`runs.json`

你用一个 JSON 列表描述每个模拟（一个 dump 对应一个 run）：

```json
[
  {
    "name": "T1100_d25_Omodel1",
    "T_K": 1100,
    "dPd_A": 25,
    "o_model": "model1",
    "o_frac": 1.0,
    "dump_path": "/ABS/PATH/TO/dump_T1100_d25_model1.lammpstrj",
    "dt_ps": 0.0004,
    "dump_every_steps": 1000,
    "stride_frames": 1
  }
]
```

说明：
- `dt_ps`: LAMMPS timestep（ps）。例如 timestep=0.4 fs -> `dt_ps=0.0004`。
- `dump_every_steps`（推荐）: 每一帧 dump 对应多少个 MD 步（例如 `dump ... 1000 ...` 则填 1000）。程序会用 `dt_frame = dt_ps * dump_every_steps` 来构造时间轴。
- `stride_frames`: 读 dump 时每隔多少帧取一帧（降采样，控数据量）
- `o_frac`（推荐）: 用一个数把“富氧→缺氧/氧空位”轴连续化，便于做回归/外推（例如 1.0=100%富氧，0.0=100%氧空位）。

---

## 3) 生成数据集（npz）

```bash
python -m zno_pd_qml.build_dataset \
  --runs_json /ABS/PATH/TO/runs.json \
  --type_map_json /ABS/PATH/TO/type_map.json \
  --out_npz /ABS/PATH/TO/dataset.npz \
  --window_ps 10 \
  --stride_ps 5 \
  --pd_surface_area_mode nominal_sphere \
  --pd_shell_thickness_A 3.0 \
  --pd_contact_cut_O_A 2.5 \
  --pd_contact_cut_Zn_A 3.0
```

输出 `dataset.npz` 包含：
- `X`: 特征矩阵（每个时间窗一个样本）
- `y`: 标签（包含 `y_cover_rate`, `y_cover_final`, `y_coop_event`, `y_coop_score`）
- `groups`: 用于 group split 的字符串（例如 `o_model` 或 `T_K` 或两者组合）
- `meta`: 每个样本对应的 run/时间窗信息（便于回溯）

---

## 4) 训练与评估（含 QML）

### 4.1 回归：预测包覆速率（推荐）
```bash
python -m zno_pd_qml.train_qml \
  --dataset_npz /ABS/PATH/TO/dataset.npz \
  --task regression \
  --target y_cover_rate \
  --group_key o_model \
  --n_qubits 4 \
  --out_dir /ABS/PATH/TO/out_train
```

### 4.2 分类：是否发生“协同攀爬事件”
```bash
python -m zno_pd_qml.train_qml \
  --dataset_npz /ABS/PATH/TO/dataset.npz \
  --task classification \
  --target y_coop_event \
  --group_key o_model \
  --n_qubits 4 \
  --out_dir /ABS/PATH/TO/out_clf
```

### 4.3 更贴近“外推”的划分方式（推荐写论文用）
- 留一氧模型外推（9折）：`--split_mode logo --group_key o_model`
- 留一温度外推：`--split_mode logo --group_key T_K`
- 留一尺寸外推：`--split_mode logo --group_key dPd_A`
- 同时外推（更难）：`--group_key o_model+T_K` 或 `o_model+dPd_A`

---

## 5) 你可以怎么写“机制/相图”

强烈建议你把“纠缠”拆成两个轴：
- **包覆轴**：`coverage`、`d(coverage)/dt`、Zn/O 到 Pd 的接触数/接触增长
- **重构轴**：Pd 形变（\(R_g\)、asphericity）、Pd-slab 接触、表面粗糙度（你也可以补充）

然后做：
- 以 \(T\)、\(d_{\mathrm{Pd}}\)、`o_model` 为条件变量
- 用模型预测“包覆速率/事件概率”
- 画出相图/边界：哪些条件更容易触发协同攀爬与快速包覆

---

## 6) 关于你现在的“加墙弛豫→去墙后势能先升后降”的建议（避免把重排与包覆纠缠）

你用的 `fix ... wall/reflect zhi 6.8` 属于“硬约束”，在去墙瞬间会让体系从**受约束势能面**切到**真实势能面**，经常出现短时间势能上冲（并不等价于“包覆在升能”）。

更稳健的做法是把“避免弛豫时包覆”改成**软约束 + 分阶段释放**：
- **软约束对象**：更推荐约束“Pd团簇整体与slab的相对位置/接触区的法向位移”，而不是反射墙。
- **释放策略**：先在约束下热化（NVT），然后逐步降低约束强度（例如每 50–100 ps 降一档），最后完全释放进入生产MD。

这样做的直接好处：
- 初态不会因为“墙的非物理反射”带入额外应力/速度分布；
- 去约束后能量曲线更平滑，你更容易把“基底/团簇重排”和“ZnO包覆”拆成两个阶段来分析与建模。


