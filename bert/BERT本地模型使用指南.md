# BERT 本地模型使用指南

## 📋 目录

- [问题背景](#问题背景)
- [快速开始](#快速开始)
- [模型转换步骤](#模型转换步骤)
- [代码变更说明](#代码变更说明)
- [常见问题](#常见问题)
- [性能优势](#性能优势)

---

## 问题背景

本地模型 `model/chinese_bert_wwm_ext_L-12_H-768_A-12` 是从 [https://github.com/ymcui/chinese-bert-wwm](https://github.com/ymcui/chinese-bert-wwm) 下载的 **TensorFlow 格式**，需要转换为 **PyTorch 格式**才能使用。

✅ **已解决**：已成功将 TensorFlow 格式的 BERT 模型转换为 PyTorch 格式，并可以正常加载使用。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**注意**：`tensorflow-cpu` 仅在首次转换模型时需要，日常使用不需要。

### 2. 测试模型加载

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

### 3. 运行情感分析

```bash
python bert/weibo_senti_bert_wwn_loaded_model.py
```

---

## 模型转换步骤

### 方法一：使用提供的转换脚本（推荐）

#### 步骤 1：安装 TensorFlow（用于模型转换）

```bash
pip install tensorflow-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 步骤 2：运行转换脚本

```bash
python bert/convert_tf_to_pytorch.py
```

该脚本会自动完成以下操作：
1. ✅ 将 TensorFlow checkpoint 转换为 PyTorch 格式（生成 `pytorch_model.bin`）
2. ✅ 创建 Hugging Face 标准配置文件（生成 `config.json`）

#### 步骤 3：运行主程序

```bash
python bert/weibo_senti_bert_wwn_loaded_model.py
```

### 方法二：手动转换

如果转换脚本遇到问题，可以手动执行以下步骤：

```python
from transformers import BertConfig, BertModel
import torch

# 加载配置
config = BertConfig.from_pretrained("model/chinese_bert_wwm_ext_L-12_H-768_A-12/bert_config.json")

# 创建模型
model = BertModel(config)

# 从 TensorFlow checkpoint 加载权重
from transformers.models.bert.convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch

convert_tf_checkpoint_to_pytorch(
    bert_config_file="model/chinese_bert_wwm_ext_L-12_H-768_A-12/bert_config.json",
    tf_checkpoint_path="model/chinese_bert_wwm_ext_L-12_H-768_A-12/bert_model.ckpt",
    pytorch_dump_path="model/chinese_bert_wwm_ext_L-12_H-768_A-12/pytorch_model.bin"
)

# 然后需要手动创建 config.json（参考 convert_tf_to_pytorch.py 中的实现）
```

### 转换后的文件结构

转换完成后，`model/chinese_bert_wwm_ext_L-12_H-768_A-12/` 目录包含以下文件：

```
model/chinese_bert_wwm_ext_L-12_H-768_A-12/
├── bert_config.json              # 原始 TensorFlow 配置（保留）
├── config.json                   # Hugging Face 标准配置（转换时自动生成）✓
├── vocab.txt                     # 词表文件
├── pytorch_model.bin             # PyTorch 模型权重（转换时自动生成）✓
└── bert_model.ckpt.*             # TensorFlow 模型（可删除）
```

### 注意事项

1. **转换只需执行一次**：转换后的 `pytorch_model.bin` 和 `config.json` 可以重复使用
2. **需要 TensorFlow**：转换过程需要安装 TensorFlow，但运行时不需要
3. **代码已更新**：[weibo_senti_bert_wwn_loaded_model.py](file://F:/project/PyTorch/bert/weibo_senti_bert_wwn_loaded_model.py) 已修改为从本地路径加载模型

### 验证转换是否成功

运行测试脚本：

```bash
python test/test_model_loading.py
```

或手动测试：

```python
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("model/chinese_bert_wwm_ext_L-12_H-768_A-12")
model = BertForSequenceClassification.from_pretrained("model/chinese_bert_wwm_ext_L-12_H-768_A-12", num_labels=2)

print("模型加载成功！")
```

---

## 代码变更说明

### 主要修改的文件

1. **[weibo_senti_bert_wwn_loaded_model.py](file://F:/project/PyTorch/bert/weibo_senti_bert_wwn_loaded_model.py)**
   - 将 `MODEL_NAME = "hfl/chinese-bert-wwm-ext"` 改为 `LOCAL_MODEL_PATH = "model/chinese_bert_wwm_ext_L-12_H-768_A-12"`
   - 从本地路径加载模型，无需网络连接

2. **[requirements.txt](file://F:/project/PyTorch/requirements.txt)**
   - 添加 `tensorflow-cpu>=2.10.0` 用于模型转换

### 新增的工具脚本

1. **[convert_tf_to_pytorch.py](file://F:/project/PyTorch/bert/convert_tf_to_pytorch.py)** - 模型转换脚本（包含自动创建 config.json）
2. **[test_model_loading.py](file://F:/project/PyTorch/test/test_model_loading.py)** - 模型加载测试脚本

---

## 常见问题

### Q: 报错 "OSError: does not appear to have a file named config.json"

**A**: 这是因为缺少 Hugging Face 标准的 `config.json` 文件。解决方法：

```bash
# 重新运行转换脚本（会自动创建 config.json）
python bert/convert_tf_to_pytorch.py
```

### Q: 是否需要一直保留 tensorflow-cpu？

**A**: 不需要。转换完成后，可以卸载 tensorflow-cpu：

```bash
pip uninstall tensorflow-cpu
```

### Q: 如何验证模型是否正确加载？

**A**: 运行测试脚本：

```bash
python test/test_model_loading.py
```

### Q: 转换失败怎么办？

**A**: 检查以下几点：
1. 确认 `bert_config.json` 和 `bert_model.ckpt.*` 文件存在
2. 确认已安装 TensorFlow：`pip show tensorflow-cpu`
3. 查看错误信息，根据提示修复

---

## 性能优势

使用本地模型的优势：

- ✅ **无需网络连接**即可加载模型
- ✅ **加载速度更快**（无需从网络下载）
- ✅ **可离线使用**，适合内网环境
- ✅ **避免网络不稳定**导致的下载失败问题
- ✅ **版本可控**，不受在线模型更新影响

---

## 🎯 下一步

模型已成功加载，现在可以：

1. **训练情感分类模型** - 运行 `weibo_senti_bert_wwn_loaded_model.py`
2. **进行文本预测** - 使用 `predict_sentiment()` 函数
3. **保存微调后的模型** - 模型会自动保存到 `bert/saved_model/` 目录

祝使用愉快！🎉
