# %% [markdown]
# # 微博情感分析 —— BERT 微调
# 基于 `bert-wwm-ext` 对微博评论进行二分类情感分析（正面 / 负面）

# %% 1. 导入依赖
import os
import random
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer


# %% 2. 超参数配置
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 5e-5
SAMPLE_SIZE = 1500
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
MODEL_NAME = "hfl/chinese-bert-wwm-ext"
DATA_PATH = "data/weibo/weibo_senti_100k.csv"
SAVE_MODEL_PATH = "bert/saved_model/bert_wwm_ext_5"

# %% 3. 加载数据
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

df = pd.read_csv(DATA_PATH)
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"数据总量: {len(df)} 条")
print(df.head())

# %% 4. 加载 Tokenizer 与模型
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
model.to(device)

# %% 5. 自定义 Dataset
class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=MAX_LENGTH):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = str(self.dataframe.iloc[idx]["review"])
        label = int(self.dataframe.iloc[idx]["label"])
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# %% 6. 划分训练集 / 验证集
dataset = SentimentDataset(df[:SAMPLE_SIZE], tokenizer)

train_size = int(TRAIN_RATIO * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集: {train_size} 条，验证集: {val_size} 条")

# %% 7. 优化器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# %% 8. 训练循环
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=correct / total)

    avg_loss = total_loss / len(train_loader)
    train_acc = correct / total
    print(f"Epoch {epoch + 1} — 平均损失: {avg_loss:.4f}，训练准确率: {train_acc:.4f}")

# %% 9. 验证评估
model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

val_accuracy = total_correct / total_samples
print(f"验证集准确率: {val_accuracy:.4f}")

# %% 10. 保存模型
model.save_pretrained(SAVE_MODEL_PATH)
tokenizer.save_pretrained(SAVE_MODEL_PATH)
print(f"模型已保存至 {SAVE_MODEL_PATH}/")

# %% 11. 情感预测函数
def predict_sentiment(sentence: str) -> str:
    """
    对单条文本进行情感预测。
    返回: '正面' 或 '负面'，以及对应概率。
    """
    model.eval()
    encoding = tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "正面" if pred == 1 else "负面"
    print(f"文本: 「{sentence}」 → {label}（置信度: {confidence:.2%}）")
    return label

# %% 12. 预测示例
predict_sentiment("气死我了")
predict_sentiment("今天天气真好，心情超棒！")
predict_sentiment("这个产品质量太差了，不推荐")

# %% 13. 加载已保存的模型进行预测
print("=" * 50)
print("加载已保存的模型...")
loaded_model = BertForSequenceClassification.from_pretrained(SAVE_MODEL_PATH, num_labels=2)
loaded_tokenizer = BertTokenizer.from_pretrained(SAVE_MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)
loaded_model.eval()
print(f"模型已从 {SAVE_MODEL_PATH} 加载")

# %% 14. 使用加载的模型进行情感预测
def predict_with_loaded_model(sentence: str) -> str:
    """
    使用已加载的模型对单条文本进行情感预测。
    返回: '正面' 或 '负面'，以及对应概率。
    """
    encoding = loaded_tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = loaded_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "正面" if pred == 1 else "负面"
    print(f"文本: 「{sentence}」 → {label}（置信度: {confidence:.2%}）")
    return label



# %% 15. 使用加载的模型进行预测
print("\n使用已加载的模型进行预测：")
predict_with_loaded_model("气死我了")
predict_with_loaded_model("今天天气真好，心情超棒！")
predict_with_loaded_model("这个产品质量太差了，不推荐")
predict_with_loaded_model("这部电影太精彩了，强烈推荐！")
predict_with_loaded_model("服务态度很差，下次不会再来了")
