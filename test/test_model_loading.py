"""
测试本地 BERT 模型加载是否正常
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 修复 torch 版本元数据问题
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

from transformers import BertTokenizer, BertForSequenceClassification

LOCAL_MODEL_PATH = "model/chinese_bert_wwm_ext_L-12_H-768_A-12"

print("=" * 60)
print("测试本地 BERT 模型加载")
print("=" * 60)
print(f"\n模型路径: {LOCAL_MODEL_PATH}\n")

try:
    print("1. 加载 Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    print("   ✓ Tokenizer 加载成功")
    
    print("\n2. 加载 BERT 模型...")
    model = BertForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH, num_labels=2)
    print("   ✓ 模型加载成功")
    
    print("\n3. 测试分词...")
    test_text = "今天天气真好"
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"   文本: {test_text}")
    print(f"   Token IDs: {tokens['input_ids'].shape}")
    print("   ✓ 分词测试成功")
    
    print("\n4. 测试模型推理...")
    import torch
    with torch.no_grad():
        outputs = model(**tokens)
        print(f"   输出 logits 形状: {outputs.logits.shape}")
        print("   ✓ 模型推理测试成功")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！模型可以正常使用")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
