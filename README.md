# PyTorch BERT 中文文本情感分析项目

基于 PyTorch 和 BERT 的中文微博情感分析工具，支持本地模型离线运行，无需网络连接。

## 📋 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [模型下载指南](#模型下载指南)
- [数据集下载指南](#数据集下载指南)
- [项目结构](#项目结构)
- [使用说明](#使用说明)
- [常见问题](#常见问题)
- [依赖环境](#依赖环境)

---

## 项目简介

本项目使用 BERTopic 进行中文文本的主题建模分析，特别适用于学术文献、社交媒体等场景。主要特点：

- ✅ **完全离线运行**：所有模型均可下载到本地，无需网络连接
- ✅ **中文分词支持**：集成 jieba 分词，优化中文处理效果
- ✅ **多模型支持**：支持 BERT、RoBERTa 等多种预训练模型
- ✅ **高精度情感分析**：基于 BERT 的情感分类准确率高
- ✅ **简单易用**：提供完整的训练和预测接口

---

## 功能特性

### 核心功能

1. **情感分类**：自动判断文本的情感倾向（正面/负面）
2. **关键词提取**：为每条文本提取重要特征词
3. **批量预测**：支持单条或批量文本情感分析
4. **模型微调**：可在自定义数据集上继续训练
5. **结果导出**：支持 CSV、Excel 等格式输出

### 应用场景

- 💬 社交媒体情感挖掘
- 📊 舆情监测与分析
- 🛍️ 产品评论情感分析
- 📰 新闻情感倾向分析

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/为flag625/pytorch.git
cd pytorch
```

### 2. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 下载模型和数据集

详见下方 [模型下载指南](#模型下载指南) 和 [数据集下载指南](#数据集下载指南)

### 4. 运行示例

```bash
# BERT 情感分析示例（使用本地模型）
python bert/bert_wwm/weibo_senti_bert_wwn_loaded_model.py

# RoBERTa WWM 扩展版
python bert/RoBERTa/weibo_senti_roberta_wwn_ext.py

# RoBERTa-330M（神思-二郎神，通过 ModelScope）
python bert/RoBERTa/weibo_senti_roberta_330m.py

# 基础微博情感分析
python bert/weibo_senti.py
```

---

## 模型下载指南

本项目支持多种预训练模型进行情感分析，包括 BERT、RoBERTa 以及专门的 RoBERTa-330M 情感分析模型。

### 一、BERT 中文预训练模型（用于情感分析）

#### 模型来源

[Chinese BERT with Whole Word Masking (WWM)](https://github.com/ymcui/chinese-bert-wwm) - 由哈工大讯飞联合实验室发布

#### 下载步骤

1. **访问下载地址**

   打开浏览器访问：https://github.com/ymcui/chinese-bert-wwm
   
   或直接下载：https://drive.google.com/drive/folders/1tR7sUaWWQfmB09K0gFf2xUpjXz8YkEoH

2. **选择模型版本**

   推荐下载以下模型之一：
   
   | 模型名称 | 大小 | 说明 |
   |---------|------|------|
   | Chinese BERT wwm Ext | ~400MB | 扩展版，性能更好（推荐） |
   | Chinese BERT wwm Base | ~400MB | 基础版 |

3. **下载文件**

   点击 `chinese_wwm_ext_L-12_H-768_A-12.zip` 下载

4. **解压并放置**

   ```bash
   # 创建模型目录
   mkdir -p model/chinese_bert_wwm_ext_L-12_H-768_A-12
   
   # 解压下载的压缩包到此目录
   # Windows: 右键 → 解压到当前文件夹
   # Linux/Mac: unzip chinese_wwm_ext_L-12_H-768_A-12.zip -d model/chinese_bert_wwm_ext_L-12_H-768_A-12/
   ```

5. **转换为 PyTorch 格式（自动批量转换）**

   TensorFlow 格式的模型需要转换为 PyTorch 格式才能使用：

   ```bash
   # 运行自动转换脚本，会自动检测并转换所有未转换的模型
   python bert/convert_tf_to_pytorch.py
   ```

   **脚本功能**：
   - ✅ 自动扫描 `model/` 目录下的所有模型
   - ✅ 智能检测已转换的模型，避免重复转换
   - ✅ 支持 BERT 和 RoBERTa 模型的自动识别
   - ✅ 自动生成 Hugging Face 标准的 `config.json` 配置文件

   转换完成后会生成以下文件：
   - `pytorch_model.bin` - PyTorch 模型权重
   - `config.json` - Hugging Face 标准配置文件

   **注意**：如果自动转换失败，推荐使用下方的"替代方案"直接下载 PyTorch 格式模型。

6. **验证安装**

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

#### 替代方案：直接从 Hugging Face 下载（推荐）

如果无法访问 Google Drive 或自动转换失败，可以从 Hugging Face 直接下载预转换好的 PyTorch 格式模型（更简单快速）：

**方法一：使用 hf 命令行工具**

```bash
# 安装 huggingface_hub
pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载 BERT 模型
hf download hfl/chinese-bert-wwm-ext --local-dir model/chinese_bert_wwm_ext_L-12_H-768_A-12
```

**方法二：手动下载**

访问：https://huggingface.co/hfl/chinese-bert-wwm-ext

下载以下文件到 `model/chinese_bert_wwm_ext_L-12_H-768_A-12/` 目录：
- `config.json`
- `pytorch_model.bin` 或 `model.safetensors`
- `vocab.txt`
- `tokenizer_config.json`
- `special_tokens_map.json`



---

### 二、模型目录结构

正确的模型目录结构应该如下：

```
model/
└── chinese_bert_wwm_ext_L-12_H-768_A-12/    # BERT 中文模型
    ├── config.json                           # Hugging Face 配置
    ├── pytorch_model.bin                     # PyTorch 模型权重
    ├── vocab.txt                             # 词表
    ├── tokenizer_config.json                 # Tokenizer 配置
    └── special_tokens_map.json               # 特殊标记映射
```

---

### 三、RoBERTa-330M 情感分析模型（推荐）

#### 模型来源

[Erlangshen-RoBERTa-330M-Sentiment](https://modelscope.cn/models/Fengshenbang/Erlangshen-RoBERTa-330M-Sentiment) - 由封神榜团队发布，专门用于情感分析的 330M 参数大模型

#### 模型特点

- ✅ **330M 大参数**：比标准 BERT Base (110M) 大 3 倍，表达能力更强
- ✅ **专门优化**：针对情感分析任务进行了专门训练
- ✅ **ModelScope 支持**：国内镜像，下载速度更快
- ✅ **开箱即用**：无需转换，直接加载 PyTorch 格式

#### 下载步骤

1. **安装 ModelScope**

   ```bash
   pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. **下载模型**

   方法一：使用 Python 代码下载
   ```python
   from modelscope import snapshot_download
   model_dir = snapshot_download('Fengshenbang/Erlangshen-RoBERTa-330M-Sentiment', cache_dir='model')
   print(f"模型已下载到: {model_dir}")
   ```

   方法二：命令行下载
   ```bash
   python -c "from modelscope import snapshot_download; snapshot_download('Fengshenbang/Erlangshen-RoBERTa-330M-Sentiment', cache_dir='model')"
   ```

3. **验证安装**

   ```bash
   # 运行测试脚本
   python bert/RoBERTa/weibo_senti_roberta_330m.py
   ```

#### 目录结构

```
model/
└── Fengshenbang/
    └── Erlangshen-RoBERTa-330M-Sentiment/
        ├── config.json
        ├── pytorch_model.bin
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── vocab.txt
```

---

### 四、模型对比与选择建议

| 模型 | 参数量 | 优势 | 适用场景 |
|------|--------|------|----------|
| BERT WWM Ext | 110M | 平衡性能和速度 | 通用情感分析 |
| RoBERTa WWM Large | 330M | 更好的泛化能力 | 复杂文本分析 |
| RoBERTa-330M Sentiment | 330M | 专门优化情感分析 | 高精度需求场景 |

**选择建议**：
- 🚀 **快速原型**：使用 BERT WWM Ext
- 🎯 **生产环境**：使用 RoBERTa-330M Sentiment
- 🔬 **研究实验**：尝试所有模型对比效果

---

## 数据集下载指南

### 微博情感分析数据集（用于 BERT 训练）

#### 数据来源

Weibo Senti 100k - 中文微博情感分析数据集

#### 数据说明

- **数量**：约 10 万条微博数据
- **标注**：正面/负面情感标签
- **格式**：CSV 文件

#### 获取方式

由于版权原因，数据集不包含在 Git 仓库中。您可以通过以下方式获取：

1. **官方渠道**

   访问：https://github.com/sophonplus/ChineseNlpCorpus
   
   查找 Weibo Senti 数据集并下载

2. **自行准备**

   如果您有自己的数据集，可以按照以下格式准备：
   
   ```csv
   text,label
   "今天天气真好，心情愉快！",1
   "这个产品太差劲了，非常失望",0
   ```

3. **放置位置**

   将下载的数据集文件放入 `data/weibo/` 目录：
   
   ```
   data/weibo/
   ├── train.csv          # 训练集
   ├── dev.csv            # 验证集
   ├── test.csv           # 测试集
   └── weibo_senti_100k.csv  # 完整数据集
   ```



---

## 项目结构

```
PyTorch/
├── bert/                          # BERT & RoBERTa 相关代码
│   ├── bert_wwm/                  # BERT WWM 实现
│   │   ├── weibo_senti_bert_wwn_ext.py           # BERT WWM 扩展版
│   │   └── weibo_senti_bert_wwn_loaded_model.py  # 加载已训练模型
│   ├── RoBERTa/                   # RoBERTa 实现（新增）
│   │   ├── weibo_senti_roberta_wwn_ext.py        # RoBERTa WWM 扩展版
│   │   └── weibo_senti_roberta_330m.py           # RoBERTa-330M 情感分析（新增）
│   ├── saved_model/               # 保存的训练模型
│   │   ├── bert_base_chinese/     # BERT 基础版模型
│   │   ├── bert_wwm_ext/          # BERT WWM 扩展版模型
│   │   ├── bert_wwm_ext_loaded_5/ # 已训练的 WWM 模型
│   │   ├── roberta-wwm-ext-large_5/      # RoBERTa WWM 模型
│   │   └── roberta_330m_sentiment_5/       # RoBERTa-330M 模型（新增）
│   ├── convert_tf_to_pytorch.py   # TF→PyTorch 转换脚本
│   ├── weibo_senti.py             # 微博情感分析主程序
│   └── BERT本地模型使用指南.md     # BERT 模型使用文档
│
├── model/                         # 预训练模型（需手动下载）
│   ├── chinese_bert_wwm_ext_L-12_H-768_A-12/  # BERT 中文模型
│   └── Fengshenbang/                           # RoBERTa-330M 模型（新增）
│       └── Erlangshen-RoBERTa-330M-Sentiment/
│
├── data/                          # 数据集（需手动下载）
│   └── weibo/                     # 微博情感数据
│       ├── train.csv              # 训练集
│       ├── dev.csv                # 验证集
│       ├── test.csv               # 测试集
│       └── weibo_senti_100k.csv   # 完整数据集
│
├── test/                          # 测试代码
│   ├── test_model_loading.py      # 模型加载测试
│   └── app.py                     # Web 应用示例
│
├── templates/                     # HTML 模板
│   └── home.html
│
├── requirements.txt               # Python 依赖包
└── README.md                      # 项目说明文档
```

---

## 使用说明

### BERT 情感分析

#### 1. 使用已训练模型进行预测

```python
from bert.bert_wwm.weibo_senti_bert_wwn_loaded_model import predict_sentiment

# 预测单条文本
result = predict_sentiment("这部电影太好看了！")
print(f"情感: {result['sentiment']}, 置信度: {result['confidence']:.2f}")

# 批量预测
texts = ["今天很开心", "心情很糟糕", "一般般吧"]
for text in texts:
    result = predict_sentiment(text)
    print(f"{text} → {result['sentiment']} ({result['confidence']:.2f})")
```

#### 2. 训练新模型

```bash
# 使用 BERT WWM 扩展版训练
python bert/bert_wwm/weibo_senti_bert_wwn_ext.py

# 使用 RoBERTa WWM 扩展版训练
python bert/RoBERTa/weibo_senti_roberta_wwn_ext.py

# 使用 RoBERTa-330M 情感分析模型训练（推荐）
python bert/RoBERTa/weibo_senti_roberta_330m.py
```

#### 3. 运行测试

```bash
# 测试模型加载
python test/test_model_loading.py

# 启动 Web 应用示例
python test/app.py
```

---

## 常见问题

### Q1: 模型加载失败，提示找不到 config.json

**A**: 这是因为 TensorFlow 格式的模型缺少 Hugging Face 标准配置文件。解决方法：

```bash
python bert/convert_tf_to_pytorch.py
```

详见 [模型下载指南](#一bert-中文预训练模型用于情感分析) 第五步。

---

### Q2: 运行时提示缺少 tf-keras

**A**: transformers 库需要 tf-keras 而非 Keras 3。解决方法：

```bash
pip uninstall keras -y
pip install tf-keras -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### Q3: 数据集太大，Git 无法推送

**A**: 数据集和模型文件不应提交到 Git。已配置 `.gitignore` 自动排除。如需共享数据，建议使用：
- Google Drive / OneDrive
- 百度网盘
- Hugging Face Datasets

---

### Q4: 如何更换为其他 BERT 模型？

**A**: 修改代码中的模型路径即可：

```python
# 例如更换为 RoBERTa
LOCAL_MODEL_PATH = "model/chinese_roberta_wwm_ext"

# 或使用 RoBERTa-330M（通过 ModelScope）
MODEL_NAME = "Fengshenbang/Erlangshen-RoBERTa-330M-Sentiment"
LOCAL_MODEL_PATH = "model/Fengshenbang/Erlangshen-RoBERTa-330M-Sentiment"
```

确保新模型已正确下载并转换为 PyTorch 格式。

---

### Q5: ModelScope 和 Hugging Face 有什么区别？

**A**: 
- **ModelScope**：阿里达摩院推出的模型平台，国内访问速度快，适合下载中文模型
- **Hugging Face**：国际知名模型平台，模型种类更全
- **推荐使用**：中文模型优先使用 ModelScope，其他模型使用 Hugging Face

---

## 依赖环境

### 系统要求

- **操作系统**：Windows 10/11, Linux, macOS
- **Python 版本**：3.9 - 3.11
- **内存**：建议 8GB 以上
- **存储**：至少 5GB 可用空间（用于模型）

### Python 依赖包

详见 [requirements.txt](requirements.txt)，主要依赖：

```txt
pandas==3.0.3
torch==2.9.1+cpu
transformers>=4.41.0,<5.0.0
jieba
matplotlib
plotly
openpyxl
```

安装命令：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 📝 许可证

本项目仅供学习和研究使用。使用的预训练模型遵循其各自的许可证协议。

---

## 🙏 致谢

- [Chinese BERT WWM](https://github.com/ymcui/chinese-bert-wwm) - 哈工大讯飞联合实验室
- [Erlangshen-RoBERTa-330M-Sentiment](https://modelscope.cn/models/Fengshenbang/Erlangshen-RoBERTa-330M-Sentiment) - 封神榜团队
- [ModelScope](https://modelscope.cn/) - 阿里达摩院模型平台
- [BERTopic](https://github.com/MaartenGr/BERTopic) - Maarten Grootendorp
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) - UKP Lab
- [Jieba](https://github.com/fxsjy/jieba) - 中文分词工具

---

## 📧 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

**GitHub**: https://github.com/flag625/pytorch

---

**祝使用愉快！** 🎉
