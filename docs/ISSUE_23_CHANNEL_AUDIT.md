# GitHub Issue #23：EngineMT-QA 32/33 通道审计

审计日期：2026-07-19

## 结论

1. 当前公开发布的 `time_series_data.h5` 中，`seq_data` 的实际 shape 是
   `(118921, 600, 33)`；对该文件而言，代码可见的通道数是 **33**。
2. H5 根节点、`seq_data` 和 `data_ID` 的 attrs 均为空；文件中也没有
   channel names、feature names、column names 或等价的通道语义映射。
3. `data_ID` 的 shape 是 `(118921,)`，值严格为 `1, 2, ..., 118921`。
   因此仓库数据加载代码采用的 `seq_data[int(id) - 1]` 是一一对齐的。
4. 训练和测试 QA 中实际出现的所有 ID 均落在 `[1, 118921]` 内，并且
   `data_ID[id - 1] == id`，未发现越界或错位。
5. 公开论文正文写的是 32 channels，而当前公开 H5 和数据集卡写的是
   33 channels。公开文件足以确认“发布张量是 33 通道”，但不足以确定
   33 个位置各自的权威名称、顺序，以及论文所述 32 通道与发布文件之间
   哪一列被新增、保留或误计。本文档不对此做语义猜测。

## 复现对象

本地文件：

- `dataset/datasets/time_series_data.h5`
- `dataset/datasets/train_qa.jsonl`
- `dataset/datasets/test_qa.jsonl`

公开来源：

- GitHub Issue #23：
  `https://github.com/Pandalin98/ITFormer-ICML25/issues/23`
- EngineMT-QA 数据集：
  `https://huggingface.co/datasets/pandalin98/EngineMT-QA`
- 论文：
  `https://arxiv.org/abs/2506.20093`

Hugging Face 当前数据集 revision：

```text
ea2051a15512212976c22c1a075ec549c7d48eef
```

本地 `sha256sum` 与公开 LFS 元数据一致：

```text
time_series_data.h5  e35bfa367df18dc4365ec96b7ef8844de1dde8fc1060e6ebb40c842e270d403c
train_qa.jsonl       195e66022038a2895e4e988050937195760f33a9f92c3fbf2b1661869114c2cf
test_qa.jsonl        d9960b9f16554453a509087aab9f26263d14c6513d699ac7d9cf7a2835f13113
```

## H5 结构证据

直接读取 H5 元数据得到：

| 对象 | shape | dtype | attrs |
|---|---:|---|---|
| root | — | — | `{}` |
| `data_ID` | `(118921,)` | `int64` | `{}` |
| `seq_data` | `(118921, 600, 33)` | `float64` | `{}` |

根节点只包含：

```text
data_ID
seq_data
```

`data_ID` 审计结果：

```text
count: 118921
min: 1
max: 118921
unique: 118921
exact row order: data_ID == [1, 2, ..., 118921]
```

这证明了行索引合同：

```python
row_index = int(qa_id) - 1
assert data_ID[row_index] == int(qa_id)
```

它只证明 ID 与 H5 行的结构对齐，不证明任一通道的物理语义。

## QA ID 与 `seq_data` 对齐证据

### 训练集

| 项目 | 数值 |
|---|---:|
| JSONL 行数 | 13,291 |
| scalar-ID records | 5,141 |
| 10-ID records | 8,150 |
| ID references | 86,641 |
| unique IDs | 6,963 |
| ID 范围 | `[20, 118920]` |
| 越界 unique IDs | 0 |
| `data_ID[id - 1] != id` | 0 |

按 JSONL 中实际 human QA turn 展平后的数量：

| Stage | 数量 |
|---|---:|
| 1 | 41,128 |
| 2 | 35,987 |
| 3 | 16,300 |
| 4 | 8,150 |
| 合计 | 101,565 |

### 测试集

| 项目 | 数值 |
|---|---:|
| JSONL 行数 | 5,539 |
| scalar-ID records | 2,155 |
| 10-ID records | 3,384 |
| ID references | 35,995 |
| unique IDs | 4,015 |
| ID 范围 | `[20, 118893]` |
| 越界 unique IDs | 0 |
| `data_ID[id - 1] != id` | 0 |

按 JSONL 中实际 human QA turn 展平后的数量：

| Stage | 数量 |
|---|---:|
| 1 | 17,240 |
| 2 | 15,085 |
| 3 | 6,768 |
| 4 | 3,384 |
| 合计 | 42,477 |

### 当前加载路径

`dataset/dataset.py` 当前按以下方式取数：

- scalar ID：读取完整的 `seq_data[id - 1]`，shape 为 `(600, 33)`。
- 10-ID list：分别读取十行，每行取前 `600 // 10 = 60` 个时间点，再沿
  时间维拼接，最终 shape 仍为 `(600, 33)`。

上述行为可由代码和 shape 直接确认；每一列的名称和物理含义不能由当前
H5 元数据确认。

## 审计脚本

依赖：

```bash
python -m pip install h5py
```

训练集：

```bash
python scripts/audit_enginemt_qa.py \
  --qa dataset/datasets/train_qa.jsonl \
  --h5 dataset/datasets/time_series_data.h5
```

测试集：

```bash
python scripts/audit_enginemt_qa.py \
  --qa dataset/datasets/test_qa.jsonl \
  --h5 dataset/datasets/time_series_data.h5
```

直接复现“按论文预期 32 通道检查发布文件”的失败：

```bash
python scripts/audit_enginemt_qa.py \
  --qa dataset/datasets/test_qa.jsonl \
  --h5 dataset/datasets/time_series_data.h5 \
  --expected-channels 32
```

该命令非零退出，并报告：

```text
seq_data channel count 33 does not match expected 32
```

默认验证发布文件合同：

```text
sequence length = 600
channel count = 33
multi-cycle ID count = 10
data_ID = one-based row order
QA IDs in [1, len(seq_data)]
data_ID[id - 1] = id
```

退出码：

- `0`：结构和 ID 对齐检查通过。
- `1`：发现异常 shape、通道数/序列长度不符、QA ID 越界、`data_ID`
  错位或 QA 结构错误。
- `2`：无法执行审计，例如缺少 `h5py`、文件不存在或 H5 无法读取。

若缺少 `h5py`，脚本会明确提示安装命令，而不是只输出 Python traceback。

## 公开文件仍不能确定的内容

以下内容不能从当前 H5、QA JSONL、公开数据集卡和论文中得到一个可验证的
权威答案：

1. 33 个 H5 通道从索引 0 到 32 的完整名称和顺序。
2. 论文所述 32 通道与发布 H5 的 33 通道之间，具体是哪一列造成差异。
3. 该差异是论文笔误、导出时额外保留了一列，还是数据预处理合同发生过
   变更。
4. 是否应删除某一列以复现论文训练。没有作者提供的映射或预处理代码，
   删除任意列都会成为未经证实的语义假设。

因此，安全的工程处理是：

- 对当前发布文件按 33 通道读取；
- 不硬编码未经证实的通道名称；
- 不为了迎合论文中的 32 而自行删除列；
- 等待作者发布 channel label/config 或权威预处理说明后，再建立语义映射。
