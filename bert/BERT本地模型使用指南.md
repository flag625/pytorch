# BERT & RoBERTa 本地模型使用指南

## 📋 目录

- [快速开始](#快速开始)
- [模型自动转换](#模型自动转换)
- [手动下载 PyTorch 模型](#手动下载-pytorch-模型)
- [代码使用说明](#代码使用说明)
- [支持的模型列表](#支持的模型列表)
- [常见问题](#常见问题)
- [性能优势](#性能优势)

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 自动转换模型

脚本会自动检测 `model/` 目录下的所有 TensorFlow 格式模型，并转换为 PyTorch 格式：

```bash
python bert/convert_tf_to_pytorch.py
```

**功能特点**：
- ✅ 自动扫描 `model/` 目录下的所有模型
- ✅ 智能检测已转换的模型，避免重复转换
- ✅ 支持 BERT 和 RoBERTa 模型的自动识别
- ✅ 自动生成 Hugging Face 标准的 `config.json` 配置文件

### 3. 测试模型加载

```bash
python test/test_model_loading.py
```

预期输出：
```
✓ Tokenizer 加载成功
✓ 模型加载成功
✓ 分词测试成功
✓ 模型推理测试成功
所有测试通过！模型可以正常使用
```

### 4. 运行情感分析

```bash
# BERT WWM 扩展版
python bert/bert_wwm/weibo_senti_bert_wwn_loaded_model.py

# RoBERTa WWM 扩展版
python bert/RoBERTa/weibo_senti_roberta_wwn_ext.py

# RoBERTa-330M（神思-二郎神）
python bert/RoBERTa/weibo_senti_roberta_330m.py
```

---

## 模型自动转换

### 转换流程

运行转换脚本后，会自动执行以下步骤：

1. **扫描模型目录**：查找 `model/` 下所有包含 `bert_config.json` 的子目录
2. **检测转换状态**：检查是否已有 `pytorch_model.bin` 和 `config.json`
3. **智能转换**：只转换未完成的模型，已转换的自动跳过
4. **生成配置**：自动创建 Hugging Face 标准的 `config.json`

### 支持的模型类型

| 模型类型 | 示例目录名 | 特征 |
|---------|-----------|------|
| BERT Base | `chinese_bert_wwm_ext_L-12_H-768_A-12` | 12层，768隐藏层 |
| BERT Large | `chinese_bert_wwm_large_ext_L-24_H-1024_A-16` | 24层，1024隐藏层 |
| RoBERTa Base | `chinese_roberta_wwm_ext_L-12_H-768_A-12` | 12层，768隐藏层 |
| RoBERTa Large | `chinese_roberta_wwm_large_ext_L-24_H-1024_A-16` | 24层，1024隐藏层 |
| RoBERTa-330M | `Fengshenbang/Erlangshen-RoBERTa-330M-Sentiment` | 330M参数，专门用于情感分析 |

### 转换后的文件结构

转换完成后，模型目录包含以下文件：

```
model/chinese_bert_wwm_ext_L-12_H-768_A-12/
├── bert_config.json              # 原始 TensorFlow 配置（保留）
├── config.json                   # Hugging Face 标准配置（自动生成）✓
├── vocab.txt                     # 词表文件
├── pytorch_model.bin             # PyTorch 模型权重（自动生成）✓
└── bert_model.ckpt.*             # TensorFlow 模型（可删除）
```

### 验证转换结果

转换成功后会显示加载示例：

**对于 BERT 模型：**
```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("model/chinese_bert_wwm_ext_L-12_H-768_A-12")
model = BertModel.from_pretrained("model/chinese_bert_wwm_ext_L-12_H-768_A-12")
```

**对于 RoBERTa 模型：**
```python
from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained("model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16")
model = RobertaModel.from_pretrained("model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16")
```

---

## 手动下载 PyTorch 模型

如果自动转换失败，可以直接从 Hugging Face 或 ModelScope 下载预转换好的 PyTorch 格式模型。

### 方法一：使用 hf 命令行工具（推荐 - BERT/RoBERTa WWM）

```bash
# 安装 huggingface_hub
pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载 BERT 模型
hf download hfl/chinese-bert-wwm-ext --local-dir model/chinese_bert_wwm_ext_L-12_H-768_A-12

# 下载 RoBERTa 模型
hf download hfl/chinese-roberta-wwm-ext-large --local-dir model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
```

### 方法二：使用 ModelScope 下载（推荐 - RoBERTa-330M）

```bash
# 安装 modelscope
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载 RoBERTa-330M 情感分析模型
from modelscope import snapshot_download
model_dir = snapshot_download('Fengshenbang/Erlangshen-RoBERTa-330M-Sentiment', cache_dir='model')
print(f"模型已下载到: {model_dir}")
```

### 方法二：从 Hugging Face 网站手动下载

1. **BERT 中文模型**：https://huggingface.co/hfl/chinese-bert-wwm-ext
   
   下载以下文件到 `model/chinese_bert_wwm_ext_L-12_H-768_A-12/`：
   - `config.json`
   - `pytorch_model.bin` 或 `model.safetensors`
   - `vocab.txt`
   - `tokenizer_config.json`
   - `special_tokens_map.json`

2. **RoBERTa 中文模型**：https://huggingface.co/hfl/chinese-roberta-wwm-ext-large
   
   下载以下文件到 `model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/`：
   - `config.json`
   - `pytorch_model.bin` 或 `model.safetensors`
   - `vocab.txt`
   - `tokenizer_config.json`
   - `special_tokens_map.json`

### 注意事项

1. **转换只需执行一次**：转换后的文件可以重复使用
2. **不需要 TensorFlow**：直接下载 PyTorch 格式无需安装 TensorFlow
3. **节省时间**：下载比转换更快，推荐使用
4. **ModelScope 支持**：RoBERTa-330M 等模型可通过 ModelScope 下载，速度更快

---

## 代码使用说明

### 主要修改的文件

#### BERT 系列
1. **[weibo_senti_bert_wwn_loaded_model.py](file://F:/project/PyTorch/bert/bert_wwm/weibo_senti_bert_wwn_loaded_model.py)**
   - 从本地路径加载模型，无需网络连接
   - 支持 BERT 和 RoBERTa 模型

#### RoBERTa 系列
2. **[weibo_senti_roberta_wwn_ext.py](file://F:/project/PyTorch/bert/RoBERTa/weibo_senti_roberta_wwn_ext.py)**
   - 使用本地 BERT WWM Ext 模型进行微调
   - 位于 `bert/RoBERTa/` 目录

3. **[weibo_senti_roberta_330m.py](file://F:/project/PyTorch/bert/RoBERTa/weibo_senti_roberta_330m.py)**
   - 基于 ModelScope 的 Erlangshen-RoBERTa-330M-Sentiment 模型
   - 支持自动从 ModelScope 下载模型
   - 330M 大参数模型，情感分析效果更佳

2. **[requirements.txt](file://F:/project/PyTorch/requirements.txt)**
   - 包含所有必要的依赖包
   - 使用清华镜像源加速下载

### 新增的工具脚本

1. **[convert_tf_to_pytorch.py](file://F:/project/PyTorch/bert/convert_tf_to_pytorch.py)** - 批量模型转换脚本
2. **[test_model_loading.py](file://F:/project/PyTorch/test/test_model_loading.py)** - 模型加载测试脚本

### 加载本地模型示例

#### BERT 模型
```python
from transformers import BertTokenizer, BertForSequenceClassification

# 从本地路径加载
LOCAL_MODEL_PATH = "model/chinese_bert_wwm_ext_L-12_H-768_A-12"

tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH, num_labels=2)

print("模型加载成功！")
```

#### RoBERTa-330M 模型（通过 ModelScope）
```python
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

MODEL_NAME = "Fengshenbang/Erlangshen-RoBERTa-330M-Sentiment"
LOCAL_MODEL_PATH = "model/Fengshenbang/Erlangshen-RoBERTa-330M-Sentiment"

# 检测本地模型，不存在则下载
if not os.path.exists(LOCAL_MODEL_PATH):
    print(f"正在从 ModelScope 下载: {MODEL_NAME}...")
    model_dir = snapshot_download(MODEL_NAME, cache_dir="model")
else:
    model_dir = LOCAL_MODEL_PATH

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)

print("模型加载成功！")
```

---

## 常见问题

### Q1: 报错 "OSError: does not appear to have a file named config.json"

**A**: 缺少 Hugging Face 标准的 `config.json` 文件。解决方法：

```bash
# 重新运行转换脚本（会自动创建 config.json）
python bert/convert_tf_to_pytorch.py
```

或直接下载 PyTorch 格式模型（见上方"手动下载"部分）。

### Q2: 是否需要安装 tensorflow-cpu？

**A**: 
- **自动转换方式**：需要安装 TensorFlow 用于转换
- **手动下载方式**：不需要 TensorFlow，直接下载 PyTorch 格式即可

如果选择手动下载，可以卸载 TensorFlow：
```bash
pip uninstall tensorflow-cpu
```

### Q3: 如何验证模型是否正确加载？

**A**: 运行测试脚本：

```bash
python test/test_model_loading.py
```

### Q4: 转换失败怎么办？

**A**: 推荐直接使用手动下载方式（见上方"手动下载 PyTorch 模型"部分），更简单快速。

### Q5: 如何添加新的模型？

**A**: 
1. 将 TensorFlow 格式的模型放入 `model/` 目录下的子文件夹
2. 确保包含 `bert_config.json` 和 `bert_model.ckpt.*` 文件
3. 运行 `python bert/convert_tf_to_pytorch.py` 自动转换

**或者直接从 ModelScope/HuggingFace 下载**：
```python
# ModelScope
from modelscope import snapshot_download
model_dir = snapshot_download('Fengshenbang/Erlangshen-RoBERTa-330M-Sentiment', cache_dir='model')

# Hugging Face
from huggingface_hub import snapshot_download
model_dir = snapshot_download('hfl/chinese-bert-wwm-ext', local_dir='model/chinese_bert_wwm_ext')
```

---

## 性能优势

使用本地模型的优势：

- ✅ **无需网络连接**即可加载模型
- ✅ **加载速度更快**（无需从网络下载）
- ✅ **可离线使用**，适合内网环境
- ✅ **避免网络不稳定**导致的下载失败问题
- ✅ **版本可控**，不受在线模型更新影响
- ✅ **批量管理**，支持多个模型同时转换

---

## 🎯 下一步

模型已成功加载，现在可以：

1. **训练情感分类模型** - 运行以下任一脚本：
   - `bert/bert_wwm/weibo_senti_bert_wwn_loaded_model.py` (BERT WWM)
   - `bert/RoBERTa/weibo_senti_roberta_wwn_ext.py` (RoBERTa WWM)
   - `bert/RoBERTa/weibo_senti_roberta_330m.py` (RoBERTa-330M)
2. **进行文本预测** - 使用 `predict_sentiment()` 函数
3. **保存微调后的模型** - 模型会自动保存到 `bert/saved_model/` 目录
4. **切换不同模型** - 修改代码中的 `LOCAL_MODEL_PATH` 即可使用不同模型
5. **尝试更大模型** - RoBERTa-330M 提供更高的准确率，适合对精度要求高的场景

祝使用愉快！🎉
