# OpenUni 数据集准备计划

## 概览

本文档详细说明 OpenUni/OpenUni2 训练所需数据集的下载和组织方案。

### 数据集分类

**Pretrain 数据集（预训练）：**
1. **megalith-10m** - 10M 高质量合成图像
2. **text-to-image-2M** - 2M 文本到图像数据（含 512px 和 1024px）
3. **laion6m** - 6M 从 Laion 精选的重新标注数据
4. **cc12m-wds** - 12M Conceptual Captions 图像数据

**Finetune 数据集（微调）：**
- **BLIP3o-60k** - 60K 高质量指令跟随数据

---

## 数据集详细信息

### 1. megalith-10m

**来源：** [drawthingsai/megalith-10m](https://huggingface.co/datasets/drawthingsai/megalith-10m)

**数据量：** ~10M 图像（分辨率 1024px）

**特点：**
- 高质量 AI 生成图像
- 包含详细的文本描述
- 数据以 .tar 格式压缩存储

**存储需求：** 约 500-800 GB（解压后）

**文件结构：**
- 原始数据：多个 .tar 文件（图像）
- 标注数据：多个 .tar 文件（captions）
- 元数据：megalith10m_all.json

---

### 2. text-to-image-2M

**来源：** [jackyhate/text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M)

**数据量：**
- 2M @ 512px 分辨率
- 10K @ 1024px 分辨率

**特点：**
- 专门为文本到图像任务设计
- 包含简洁的生成导向 prompts
- 两种分辨率可选

**存储需求：** 约 300-500 GB

**文件结构：**
- data_512_2M/: 主要训练数据（多个 .tar）
- data_1024_10K/: 高分辨率数据（1个 .tar）

---

### 3. laion6m

**来源：** [dclure/laion-aesthetics-12m-umap](https://huggingface.co/datasets/dclure/laion-aesthetics-12m-umap)

**标注来源：** [wusize/laion6m_recap](https://huggingface.co/datasets/wusize/laion6m_recap)

**数据量：** ~6M 图像

**特点：**
- 从 Laion 12M 筛选的高美学质量图像
- 重新标注的 captions（使用更好的模型）
- 宽高比限制在 1:3 以内

**注意事项：**
- ⚠️ **不能直接分发图像，仅提供 URLs 和 captions**
- 需要使用 [img2dataset](https://github.com/rom1504/img2dataset) 工具下载
- 下载可能会有部分失败（链接失效）

**存储需求：** 约 400-600 GB

---

### 4. cc12m-wds (Conceptual Captions 12M)

**来源：**
- 图像：[pixparse/cc12m-wds](https://huggingface.co/datasets/pixparse/cc12m-wds)
- 标注：[wusize/cc12m_recap](https://huggingface.co/datasets/wusize/cc12m_recap)

**数据量：** ~12M 图像

**特点：**
- Google Conceptual Captions 数据集
- 重新标注为更简洁的生成导向 prompts
- 数据以 WebDataset 格式存储

**存储需求：** 约 800-1000 GB

**文件结构：**
- 多个 .tar 文件（WebDataset 格式）
- 包含图像和元数据

---

### 5. BLIP3o-60k (Finetune)

**来源：** [BLIP3o/BLIP3o-60k](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k)

**数据量：** 60K 图像

**特点：**
- 高质量的指令跟随数据
- 用于 finetune 阶段
- ⚠️ 包含 10K GenEval 相同模板的样本（可能存在数据泄露）

**存储需求：** 约 10-20 GB

**文件结构：**
- 多个 .tar 文件
- 需要解压提取

---

## 目录规划

### 推荐的目录结构

```
/gemini/code/OpenUni/
├── data/
│   ├── megalith-10m/
│   │   ├── raw/                          # 原始图像（解压后）
│   │   │   ├── 000000/
│   │   │   ├── 000001/
│   │   │   └── ...
│   │   ├── captions/                     # 标注文件（解压后）
│   │   │   ├── 000000.json
│   │   │   ├── 000001.json
│   │   │   └── ...
│   │   └── megalith10m_all.json          # 元数据索引
│   │
│   ├── text-to-image-2M/
│   │   ├── raw/
│   │   │   ├── data_512_2M/              # 2M @ 512px（解压后）
│   │   │   │   ├── data_000000/
│   │   │   │   ├── data_000001/
│   │   │   │   └── ...
│   │   │   └── data_1024_10K/            # 10K @ 1024px（解压后）
│   │   │       └── data_000000/
│   │   └── data/
│   │       ├── data_512_2M.json
│   │       └── data_1024_10K.json
│   │
│   ├── laion6m/
│   │   ├── raw/                          # 使用 img2dataset 下载
│   │   │   ├── 00000000/
│   │   │   │   ├── 00000001.jpg
│   │   │   │   ├── 00000001.json
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── parquets/                     # URLs 和 captions（parquet 格式）
│   │   │   ├── train-00000-of-00010.parquet
│   │   │   └── ...
│   │   └── data.json                     # 元数据索引
│   │
│   ├── cc12m/
│   │   ├── raw/                          # 图像（WebDataset 解压后）
│   │   │   ├── 00000/
│   │   │   ├── 00001/
│   │   │   └── ...
│   │   ├── captions/                     # 重新标注的 captions
│   │   │   ├── 00000.json
│   │   │   └── ...
│   │   └── data.json                     # 元数据索引
│   │
│   └── BLIP3o-60k/
│       ├── raw/                          # 原始数据（解压后）
│       │   ├── 000000/
│       │   ├── 000001/
│       │   └── ...
│       └── data.json                     # 元数据索引
```

---

## 存储空间需求总结

| 数据集 | 压缩大小 (估算) | 解压后大小 (估算) | 优先级 |
|--------|----------------|------------------|-------|
| megalith-10m | ~200 GB | ~500-800 GB | 高 |
| text-to-image-2M | ~100 GB | ~300-500 GB | 高 |
| laion6m | N/A (动态下载) | ~400-600 GB | 中 |
| cc12m-wds | ~300 GB | ~800-1000 GB | 中 |
| BLIP3o-60k | ~5 GB | ~10-20 GB | 高 (Finetune) |
| **总计** | **~605+ GB** | **~2.0-2.9 TB** | - |

**建议：**
- 最小配置（仅 pretrain 核心）：megalith-10m + text-to-image-2M = ~1.2 TB
- 完整 pretrain：以上 + laion6m + cc12m = ~2.5 TB
- Finetune：BLIP3o-60k = ~20 GB

---

## 下载策略

### 阶段 1：优先下载（必需数据）

1. **BLIP3o-60k** (Finetune 必需)
   - 数据量小，快速下载
   - 可先进行 finetune 实验

2. **text-to-image-2M**
   - 高质量数据
   - 文件数适中，下载较快

3. **megalith-10m**
   - 大规模高质量数据
   - 作为主要 pretrain 数据源

### 阶段 2：扩展下载（可选）

4. **cc12m-wds**
   - 增加数据多样性
   - 文件数多，需要时间

5. **laion6m**
   - 需要使用 img2dataset 工具
   - 下载可能不稳定
   - 可能有失败的 URLs

---

## 数据处理流程

### 标准处理流程（以 megalith-10m 为例）

1. **下载原始数据**
   ```bash
   huggingface-cli download drawthingsai/megalith-10m \
     --local-dir data/megalith-10m/raw --repo-type dataset
   ```

2. **下载元数据/标注**
   ```bash
   huggingface-cli download wusize/megalith-10m \
     --local-dir data/megalith-10m --repo-type dataset
   ```

3. **解压图像**
   ```bash
   cd data/megalith-10m/raw
   python extract.py --num-processes 8
   ```

4. **解压标注**
   ```bash
   cd data/megalith-10m/captions
   python extract.py --num-processes 8
   ```

5. **验证数据完整性**
   - 检查解压后的文件数量
   - 验证 JSON 格式正确性

### 特殊情况：laion6m

由于版权限制，laion6m 需要特殊处理：

1. **下载 URLs 和 captions**
   ```bash
   huggingface-cli download wusize/laion6m_recap \
     --local-dir data/laion6m/parquets --repo-type dataset
   ```

2. **使用 img2dataset 下载图像**
   ```bash
   # 需要先安装 img2dataset
   pip install img2dataset

   # 从 parquet 文件下载图像
   img2dataset --url_list data/laion6m/parquets \
               --input_format "parquet" \
               --output_folder data/laion6m/raw \
               --processes_count 16 \
               --thread_count 64 \
               --image_size 512 \
               --resize_mode keep_ratio \
               --min_image_size 256
   ```

3. **生成数据索引**
   - 根据下载成功的图像生成 data.json

---

## 并行下载建议

由于数据量巨大，建议采用以下策略：

### 方案 A：串行下载（稳定但慢）
```bash
# 按优先级顺序下载
1. BLIP3o-60k
2. text-to-image-2M
3. megalith-10m
4. cc12m-wds
5. laion6m
```

### 方案 B：并行下载（快但占用资源）
```bash
# 同时启动多个下载任务（不同后台进程）
# 需要注意网络带宽和磁盘 I/O
```

### 方案 C：分批下载（推荐）
```bash
# 批次 1（小文件，测试用）
- BLIP3o-60k

# 批次 2（核心数据）
- text-to-image-2M
- megalith-10m

# 批次 3（扩展数据）
- cc12m-wds
- laion6m（需要额外工具）
```

---

## 数据完整性检查

下载完成后，建议进行以下检查：

1. **文件数量验证**
   ```bash
   # 检查 .tar 文件数量是否匹配
   ls data/megalith-10m/raw/*.tar | wc -l
   ```

2. **解压后文件验证**
   ```bash
   # 检查解压后的图像数量
   find data/megalith-10m/raw -name "*.jpg" | wc -l
   ```

3. **JSON 格式验证**
   ```python
   import json
   with open('data/megalith-10m/megalith10m_all.json') as f:
       data = json.load(f)
       print(f"Total samples: {len(data)}")
   ```

4. **随机抽样检查**
   - 随机选择 100 个样本
   - 验证图像可以正常加载
   - 验证标注文件存在且格式正确

---

## 预计时间估算

基于 1 Gbps 网络和 SSD 存储：

| 数据集 | 下载时间 | 解压时间 | 总时间 |
|--------|---------|---------|--------|
| BLIP3o-60k | 10-30 分钟 | 10-20 分钟 | ~1 小时 |
| text-to-image-2M | 3-5 小时 | 1-2 小时 | ~6 小时 |
| megalith-10m | 6-10 小时 | 2-4 小时 | ~12 小时 |
| cc12m-wds | 10-15 小时 | 3-5 小时 | ~18 小时 |
| laion6m | 15-24+ 小时 | 不适用 | ~24+ 小时 |

**总计：** 约 2-4 天（如果串行下载）

**并行下载：** 可缩短至 1-2 天（取决于带宽和存储性能）

---

## 注意事项

1. **磁盘空间监控**
   - 下载过程中持续监控磁盘空间
   - 建议预留 20% 额外空间用于解压和临时文件

2. **网络稳定性**
   - 大文件下载建议使用 `--resume-download` 参数
   - 可能需要多次重试失败的下载

3. **HuggingFace Token**
   - 部分数据集可能需要登录
   - 使用 `huggingface-cli login` 提前认证

4. **版权和使用许可**
   - 确保遵守各数据集的使用许可
   - laion6m 和 cc12m 仅用于研究目的

5. **数据质量**
   - laion6m 可能有部分损坏或无法下载的图像
   - 需要过滤和清洗

---

## 下一步计划

1. ✅ 阅读和理解所有数据集文档
2. ✅ 规划目录结构和存储分配
3. ⏳ 编写自动化下载脚本（见 `download_datasets.sh`）
4. ⏳ 开始下载数据（按优先级）
5. ⏳ 数据验证和完整性检查
6. ⏳ 准备训练配置文件

---

## 相关资源

- [OpenUni 官方文档](https://github.com/wusize/OpenUni)
- [img2dataset 工具](https://github.com/rom1504/img2dataset)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
