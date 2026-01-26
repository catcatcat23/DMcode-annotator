

---

# DMcode-annotator

> ECC200 标准下重建/修复损坏的 Data Matrix (DM) 码。([GitHub][1])

本项目聚焦于 **Data Matrix ECC200** 的“可视化编辑 + 编解码验证 + 批处理流程”——用于在 DM 码存在缺损、污染、局部遮挡时，辅助你把**网格（module/cell）层面的信息**整理出来，并尝试重建/验证可解码性。

---

## 你能用它做什么

* **网格级别编辑/修复**：对 DM 码的 cell（黑/白模块）做人工/半自动调整（见 `dmcode_cell_editor.py`）。([GitHub][1])
* **ECC200 编码/解码验证**：把重建后的网格/图像进行编码或解码测试（见 `encode_decode.py`、`zxing_test.py`）。([GitHub][1])
* **批处理流水线**：对一批图像/结果做整理、筛选、差分、同步等工程化操作（如 `dmcode_pipeline_robin.py`、`collect_model_fail.py`、`make_diff.py`、`copy_sync_checked.py` 等）。([GitHub][1])
* **Web/本地应用入口（若你写的是界面）**：仓库里有 `app.py` / `dev_app.py`，通常用于启动交互式页面或调试入口。([GitHub][1])

---

## 背景：为什么 ECC200 能“修复”

Data Matrix **ECC200** 使用 **Reed–Solomon** 纠错码，在符号部分受损时仍可能恢复出完整数据；许多介绍资料都会提到 ECC200 对“受损区域”的恢复能力（常见说法是可在一定比例损坏下仍有机会重建/解码）。([Cognex][2])

> 注意：能否成功不仅取决于损坏比例，也取决于定位/采样精度、印刷质量、畸变、对比度、噪声与解码器实现。

---

## 项目结构（按当前仓库文件）

> 以仓库首页现有文件为准：([GitHub][1])

* `app.py` / `dev_app.py`：应用入口/调试入口（启动界面或本地服务）
* `encode_decode.py`：编码/解码流程与验证
* `dmcode_pipeline_robin.py`：批处理/流水线脚本
* `zxing_test.py`：ZXing 解码测试
* `collect_model_fail.py` / `make_diff.py` / `copy_sync_checked.py` / `crop_ori_with_checkedcases.py` / `find_ori_image.py`：工程辅助脚本
* `SNcode/`：数据与中间产物目录（你自己的工程目录）

---

## 环境要求

建议（按常见 DM 图像处理栈）：

* Python 3.9+
* `numpy`
* `opencv-python`
* `Pillow`
* （可选）`pyzxing` / `zxing-cpp` 或其他 ZXing 绑定（用于对照解码）
* （可选）你常用的 DM 解码库：`pylibdmtx` / `libdmtx`（用于对照验证）


## 安装

```bash
git clone https://github.com/catcatcat23/DMcode-annotator.git
cd DMcode-annotator

# 建议新建虚拟环境
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows PowerShell

pip install -U pip
pip install numpy opencv-python pillow
# 按需安装 zxing / pylibdmtx 等
```


---

# DMcode-annotator

在 **ECC200** 标准下，对损坏/缺失的 Data Matrix (DM) 码进行 **自动重建 + 解码验证 + 标注(mask) + 手工修复** 的工具仓库。

## 项目整体思路

本项目采用“**先自动、再标注、最后人工兜底**”的工作流：

1. **先跑批处理脚本**：自动执行 encode/decode 和结果整理，产出可解码与不可解码样本（badcases）。
2. **再启动 `app.py`**：对“成功解码 / 未解码”的数据进行 **mask 标注**（用于后续分析/训练/修复）。
3. **如果自动脚本效果较差（badcases 很多）**：启动 `dev_app.py`，手动设置网格并手动画 DM 码（人工修复兜底）。

---

## SNcode 目录说明（重要）

`SNcode/` 是本项目的数据工作目录，用于存放 DM 数据以及脚本生成的中间结果/检查结果。典型包含：

* **原始/待处理 DM 数据**：图片、csv、同步后的文件等
* **脚本运行结果**：如 encode/decode 生成的分类结果
* **badcases**：批处理后 **无法解码或需要进一步处理** 的样本集合（后续重点标注/修复对象）
* **checkedcases**：人工确认/复核过的样本（如果你有该流程）


---

## 推荐工作流（你现在的标准流程）

### Step 1：先跑 encode/decode 批处理（Robin 版本）

先运行批处理脚本（你说的 *encode decode robin*），它会：

* 在 ECC200 逻辑下尝试重建/编码/解码
* 把结果按“可解码/不可解码”等规则整理
* 输出到 `SNcode` 下，并生成 `SNcode/badcases`（重点关注）

示例（按你仓库实际脚本名，命令以你本地参数为准）：

```bash

python dmcode_pipeline_robin.py
```

运行完成后，你应该能在 `SNcode/` 下看到整理后的结果，其中 `SNcode/badcases/` 是下一步要重点处理的内容。

---

### Step 2：启动 `app.py` 做 mask 标注

当 Step 1 产出结果后，启动应用对数据做标注：

* 对 **成功解码** 的数据标注 mask（例如定位区域/辅助信息）
* 对 **未解码（badcases）** 的数据标注 mask（通常更关键，用于分析失败原因/后续修复）

启动方式：

```bash
python app.py
```

---

### Step 3：如果自动处理效果差，启动 `dev_app.py` 进行手工修复

当你发现脚本处理效果较差，比如：

* `badcases` 数量很多
* 自动推断的网格/对齐不稳定
* 自动重建后仍无法解码

你可以进入 **dev 模式**：

* **手动设置网格（grid）**
* **手动画 DM 码（cell/模块级）**
* 用人工方式把码“修回可解码”或至少修到可用于标注/训练

启动方式：

```bash
python dev_app.py
```

---

## 其他文件说明

仓库中除了上述主流程文件外，其余脚本主要是 **数据处理/工程辅助脚本**，包括但不限于：

* 样本收集、筛选、同步：`collect_model_fail.py`、`copy_sync_checked.py`
* 原图定位/裁剪：`find_ori_image.py`、`crop_ori_with_checkedcases.py`
* 差分/对比：`make_diff.py`
* 结果整理/拆分：`split_results_by_prefix.py`、`split_by_bbox_ratio.py`
* 清理/过滤：`rm_checked_pad.py`
* 解码对照测试：`zxing_test.py`




## 致谢（可选）

* Data Matrix ECC200 的纠错基于 Reed–Solomon；ECC200 的鲁棒性是本项目“可修复”的基础。([Cognex][2])

---

## License

（如果你还没选许可证，可以先写 `TBD`，或者补一个 MIT/Apache-2.0）

---

如果你把 `app.py` / `dmcode_cell_editor.py` 的“实际启动方式”和“输入输出格式”（比如：读一张图还是读一个目录、输出保存在哪里、导出网格用什么格式）贴我一段，我可以把 README 里“快速开始”那部分改成**完全可复制运行**的版本，并把参数表也补齐。

[1]: https://github.com/catcatcat23/DMcode-annotator "GitHub - catcatcat23/DMcode-annotator: ecc200标准下重建损坏dm码"
[2]: https://www.cognex.com/resources/symbologies/2-d-matrix-codes/data-matrix-codes?utm_source=chatgpt.com "Data Matrix Codes - Symbologies"
