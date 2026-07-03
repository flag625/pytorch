"""
将 TensorFlow BERT 模型转换为 PyTorch 格式
"""
import os
import sys

# 修复 torch 版本元数据问题
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import importlib.metadata as _metadata
if not getattr(_metadata.version, '_torch_patched', False):
    _original = _metadata.version
    def _safe_metadata_version(name):
        result = _safe_metadata_version._torch_original(name)
        if result is None and name.lower() == 'torch':
            import torch as _torch
            return _torch.__version__.split('+')[0]
        return result
    _safe_metadata_version._torch_original = _original
    _safe_metadata_version._torch_patched = True
    _metadata.version = _safe_metadata_version

from transformers.models.bert.convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch

# 配置路径
TF_CHECKPOINT_DIR = "model/chinese_bert_wwm_ext_L-12_H-768_A-12"
BERT_CONFIG_FILE = os.path.join(TF_CHECKPOINT_DIR, "bert_config.json")
TF_CHECKPOINT_FILE = os.path.join(TF_CHECKPOINT_DIR, "bert_model.ckpt")
PYTORCH_DUMP_OUTPUT = os.path.join(TF_CHECKPOINT_DIR, "pytorch_model.bin")

print("=" * 60)
print("BERT TensorFlow 到 PyTorch 模型转换工具")
print("=" * 60)
print(f"\n配置文件: {BERT_CONFIG_FILE}")
print(f"TF checkpoint: {TF_CHECKPOINT_FILE}")
print(f"PyTorch 输出: {PYTORCH_DUMP_OUTPUT}\n")

# 检查文件是否存在
if not os.path.exists(BERT_CONFIG_FILE):
    print(f"错误: 配置文件不存在 - {BERT_CONFIG_FILE}")
    sys.exit(1)

if not os.path.exists(TF_CHECKPOINT_FILE + ".index"):
    print(f"错误: TF checkpoint 文件不存在 - {TF_CHECKPOINT_FILE}")
    sys.exit(1)

try:
    print("开始转换...")
    convert_tf_checkpoint_to_pytorch(
        bert_config_file=BERT_CONFIG_FILE,
        tf_checkpoint_path=TF_CHECKPOINT_FILE,
        pytorch_dump_path=PYTORCH_DUMP_OUTPUT
    )
    print("\n✓ 模型转换成功!")
    print(f"PyTorch 模型已保存至: {PYTORCH_DUMP_OUTPUT}")
    
    # 创建标准的 config.json 文件
    print("\n正在创建 Hugging Face 标准配置文件...")
    import json
    with open(BERT_CONFIG_FILE, 'r', encoding='utf-8') as f:
        bert_config = json.load(f)
    
    hf_config = {
        "architectures": ["BertForSequenceClassification"],
        "attention_probs_dropout_prob": bert_config.get("attention_probs_dropout_prob", 0.1),
        "classifier_dropout": None,
        "directionality": "bidi",
        "hidden_act": bert_config.get("hidden_act", "gelu"),
        "hidden_dropout_prob": bert_config.get("hidden_dropout_prob", 0.1),
        "hidden_size": bert_config.get("hidden_size", 768),
        "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
        "initializer_range": bert_config.get("initializer_range", 0.02),
        "intermediate_size": bert_config.get("intermediate_size", 3072),
        "label2id": {"LABEL_0": 0, "LABEL_1": 1},
        "layer_norm_eps": bert_config.get("layer_norm_eps", 1e-12),
        "max_position_embeddings": bert_config.get("max_position_embeddings", 512),
        "model_type": "bert",
        "num_attention_heads": bert_config.get("num_attention_heads", 12),
        "num_hidden_layers": bert_config.get("num_hidden_layers", 12),
        "pad_token_id": bert_config.get("pad_token_id", 0),
        "pooler_fc_size": bert_config.get("pooler_fc_size", 768),
        "pooler_num_attention_heads": bert_config.get("pooler_num_attention_heads", 12),
        "pooler_num_fc_layers": bert_config.get("pooler_num_fc_layers", 3),
        "pooler_size_per_head": bert_config.get("pooler_size_per_head", 128),
        "pooler_type": bert_config.get("pooler_type", "first_token_transform"),
        "position_embedding_type": "absolute",
        "transformers_version": "4.40.0",
        "type_vocab_size": bert_config.get("type_vocab_size", 2),
        "use_cache": True,
        "vocab_size": bert_config.get("vocab_size", 21128)
    }
    
    CONFIG_JSON_PATH = os.path.join(TF_CHECKPOINT_DIR, "config.json")
    with open(CONFIG_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(hf_config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 已创建标准配置文件: {CONFIG_JSON_PATH}")
    print("\n现在可以使用以下代码加载模型:")
    print(f'  tokenizer = BertTokenizer.from_pretrained("{TF_CHECKPOINT_DIR}")')
    print(f'  model = BertForSequenceClassification.from_pretrained("{TF_CHECKPOINT_DIR}", num_labels=2)')
except Exception as e:
    print(f"\n✗ 转换失败: {e}")
    sys.exit(1)
