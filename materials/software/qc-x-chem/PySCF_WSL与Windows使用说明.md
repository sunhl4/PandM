# PySCF：在 Ubuntu / WSL 安装后，如何在 Windows 侧使用

本文说明：按 [`README.md`](README.md) 在 **Ubuntu 或 WSL2** 中装好 PySCF 之后，日常在 **Windows** 上写代码、跑 Notebook 时应如何调用——以及**不能**怎么做。

## 核心结论

- 在 Ubuntu / WSL 里安装的 PySCF，属于 **Linux 版 Python 环境**（Linux 动态库 + 该环境下的解释器）。
- **无法**在 **Windows 原生** 的 `python.exe`（例如 Anaconda 的 `win-64` 环境 `qc_chem`）里直接 `import` 到 WSL/Ubuntu 里的那份 PySCF。两套操作系统、两套解释器，**库不能混用**。
- 要在「Windows 桌面 + 已装好的 PySCF」上工作，正确做法是：**让 Python 在 WSL/Linux 里运行**；浏览器、编辑器、终端可以仍在 Windows 上。

## 推荐方式一：在 WSL 终端里跑 Python

1. 打开 **WSL（Ubuntu）** 终端。
2. 进入工程目录（示例，按你的盘符与路径修改）：
   ```bash
   cd /mnt/d/Yaozheng/PandM/materials/software/qc-x-chem
   ```
3. 激活 Conda 环境（若 Conda 安装在 WSL 内）：
   ```bash
   conda activate qc_chem
   ```
4. 运行脚本或 REPL：
   ```bash
   python your_script.py
   python -c "import pyscf; print(pyscf.__version__)"
   ```

## 推荐方式二：从 Windows PowerShell 调用 WSL 里的 Python

不进入交互式 WSL 时，可用 `wsl` 执行一条命令（需把 Conda 初始化路径改成你在 WSL 中的实际路径）：

```powershell
wsl -e bash -lc "source ~/anaconda3/etc/profile.d/conda.sh && conda activate qc_chem && cd /mnt/d/Yaozheng/PandM/materials/software/qc-x-chem && python your_script.py"
```

说明：

- `source .../conda.sh`：若你用的是 Miniconda/Anaconda 默认安装位置，常见为 `~/anaconda3` 或 `~/miniconda3`。
- `cd` 后的路径必须是 **WSL 路径**（如 `/mnt/d/...`），不要用 `D:\...`。

## 推荐方式三：Cursor / VS Code「在 WSL 中打开」

1. 安装 **WSL** 相关扩展（如 *WSL*）。
2. 使用 **在 WSL 中打开文件夹**（或 `\\wsl$\Ubuntu\mnt\d\Yaozheng\...`）。
3. 在编辑器里把终端、调试目标、Jupyter 内核选为 **WSL 内的 Python（如 `qc_chem`）**。

这样 `import pyscf` 使用的始终是 Linux 侧已安装的环境。

## Jupyter：在 WSL 里启动，在 Windows 浏览器里用

在 WSL 中：

```bash
conda activate qc_chem
cd /mnt/d/Yaozheng/PandM/materials/software/qc-x-chem
jupyter lab --no-browser
```

终端里会打印带 `token` 的本地 URL。在 **Windows** 的浏览器中打开 `http://localhost:8888`（端口以实际输出为准）。计算仍在 WSL 内执行，PySCF 可用。

## 若必须使用「纯 Windows 解释器」

若运行内核必须是 **原生 Windows Python**（不经过 WSL），则 **WSL/Ubuntu 里安装的 PySCF 无法被该解释器加载**。需要在 Windows 的 `qc_chem` 中**单独**安装 Windows 可用的 PySCF，参见 [`README.md`](README.md) 中的 **Windows 说明**与 `install_pyscf_windows.py`。

## 远程 Ubuntu 物理机 / 虚拟机

若 PySCF 装在**另一台 Linux 主机**上而非本机 WSL，则通常通过 **SSH** 登录该主机运行 Python/Jupyter，或用远程开发（Remote SSH）打开远程工程；同样不是把 Linux 的 `site-packages` 挂到 Windows Python 上使用。

## 相关文件

- 环境一览与安装脚本入口：[`README.md`](README.md)
- WSL 安装脚本：`install_pyscf_wsl.sh`、`install_pyscf_wsl.ps1`
- 原生 Windows 修补安装：`install_pyscf_windows.py`
