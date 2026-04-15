# 外部符号与定义锚点（阶段 3）

仓库内推导见 [`quantum_chemistry/docs/`](../quantum_chemistry/docs/)。**对外写作或论文级符号**时，以下二者之一为“主锚”，避免与半经典/不同软件约定混用。

## A. 电子结构教材（选一本为主）

| 资源 | 用途 |
|------|------|
| Szabo & Ostlund, *Modern Quantum Chemistry* | Hartree–Fock、CI、MP、CC 入门符号与能量分解 |
| Helgaker et al., *Molecular Electronic-Structure Theory* | 更完整的数学与 MCSCF/CC 参考 |

**建议**：在 Zotero 建一条“主参考书”，全文引用以该书章节为准。

更长的教材与参考书列表（含 Messiah、Shankar、Jensen 等）见 [`literature/references/textbooks.md`](literature/references/textbooks.md)。

## B. 量子软件栈（与实现一致）

按你实际代码选择 **其一** 作为 API/符号主锚：

| 栈 | 文档入口 |
|----|----------|
| **PennyLane** | [Quantum Chemistry](https://docs.pennylane.ai/en/stable/introduction/chemistry.html)、[`qml.qchem`](https://docs.pennylane.ai/en/stable/code/qml_qchem.html) |
| **Qiskit Nature** | [Documentation](https://qiskit-community.github.io/qiskit-nature/)（哈密顿量、VQE、激发态模块） |

若仓库教程同时使用两者，请在内部报告 **§符号表** 中写明：例如“哈密顿量由 PySCF 导出，VQE 在 PennyLane 中实现”。

## C. arXiv / 期刊

预印本以 **arXiv ID + 版本日期** 为准；正式引用以 **期刊 DOI** 为准。Zotero 用 `arxiv:` 标识符避免重复条目。
