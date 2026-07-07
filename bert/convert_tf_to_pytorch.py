"""
批量转换 TensorFlow BERT/RoBERTa 模型为 PyTorch 格式
自动检测 model/ 目录下的所有模型，只转换未完成的模型
"""
import os
import sys
import json
import subprocess
from pathlib import Path


def check_model_converted(model_dir):
    """检查模型是否已经转换为 PyTorch 格式"""
    pytorch_model = os.path.join(model_dir, "pytorch_model.bin")
    config_json = os.path.join(model_dir, "config.json")
    
    # 必须同时存在 pytorch_model.bin 和 config.json
    return os.path.exists(pytorch_model) and os.path.exists(config_json)


def detect_model_type(bert_config):
    """根据配置检测模型类型（BERT 或 RoBERTa）"""
    num_hidden_layers = bert_config.get("num_hidden_layers", 12)
    hidden_size = bert_config.get("hidden_size", 768)
    
    # RoBERTa 通常层数更多或隐藏层更大
    is_roberta = num_hidden_layers > 12 or hidden_size > 768
    return "roberta" if is_roberta else "bert"


def create_hf_config(bert_config, model_type="bert"):
    """创建 Hugging Face 格式的 config.json"""
    num_hidden_layers = bert_config.get("num_hidden_layers", 12)
    hidden_size = bert_config.get("hidden_size", 768)
    
    if model_type == "roberta":
        # RoBERTa 配置
        hf_config = {
            "architectures": ["RobertaModel"],
            "attention_probs_dropout_prob": bert_config.get("attention_probs_dropout_prob", 0.1),
            "bos_token_id": 0,
            "classifier_dropout": None,
            "eos_token_id": 2,
            "hidden_act": bert_config.get("hidden_act", "gelu"),
            "hidden_dropout_prob": bert_config.get("hidden_dropout_prob", 0.1),
            "hidden_size": hidden_size,
            "initializer_range": bert_config.get("initializer_range", 0.02),
            "intermediate_size": bert_config.get("intermediate_size", 4096),
            "layer_norm_eps": bert_config.get("layer_norm_eps", 1e-5),
            "max_position_embeddings": bert_config.get("max_position_embeddings", 514),
            "model_type": "roberta",
            "num_attention_heads": bert_config.get("num_attention_heads", 16),
            "num_hidden_layers": num_hidden_layers,
            "pad_token_id": 1,
            "position_embedding_type": "absolute",
            "transformers_version": "4.40.0",
            "type_vocab_size": bert_config.get("type_vocab_size", 2),
            "use_cache": True,
            "vocab_size": bert_config.get("vocab_size", 21128)
        }
    else:
        # BERT 配置
        hf_config = {
            "architectures": ["BertModel"],
            "attention_probs_dropout_prob": bert_config.get("attention_probs_dropout_prob", 0.1),
            "classifier_dropout": None,
            "directionality": "bidi",
            "hidden_act": bert_config.get("hidden_act", "gelu"),
            "hidden_dropout_prob": bert_config.get("hidden_dropout_prob", 0.1),
            "hidden_size": hidden_size,
            "initializer_range": bert_config.get("initializer_range", 0.02),
            "intermediate_size": bert_config.get("intermediate_size", 3072),
            "layer_norm_eps": bert_config.get("layer_norm_eps", 1e-12),
            "max_position_embeddings": bert_config.get("max_position_embeddings", 512),
            "model_type": "bert",
            "num_attention_heads": bert_config.get("num_attention_heads", 12),
            "num_hidden_layers": num_hidden_layers,
            "pad_token_id": bert_config.get("pad_token_id", 0),
            "position_embedding_type": "absolute",
            "transformers_version": "4.40.0",
            "type_vocab_size": bert_config.get("type_vocab_size", 2),
            "use_cache": True,
            "vocab_size": bert_config.get("vocab_size", 21128)
        }
    
    return hf_config


def convert_single_model(model_dir, model_name):
    """转换单个模型"""
    print(f"\n{'='*70}")
    print(f"处理模型: {model_name}")
    print(f"{'='*70}")
    
    # 检查是否已转换
    if check_model_converted(model_dir):
        print(f"✓ 模型 '{model_name}' 已经转换完成，跳过")
        return True
    
    # 检查必要的文件是否存在
    bert_config_file = os.path.join(model_dir, "bert_config.json")
    tf_checkpoint = os.path.join(model_dir, "bert_model.ckpt")
    
    if not os.path.exists(bert_config_file):
        print(f"✗ 错误: 找不到配置文件 {bert_config_file}")
        return False
    
    if not os.path.exists(tf_checkpoint + ".index"):
        print(f"✗ 错误: 找不到 TensorFlow checkpoint 文件 {tf_checkpoint}")
        return False
    
    print(f"配置文件: {bert_config_file}")
    print(f"TF checkpoint: {tf_checkpoint}")
    
    # 读取配置并检测模型类型
    try:
        with open(bert_config_file, 'r', encoding='utf-8') as f:
            bert_config = json.load(f)
        
        model_type = detect_model_type(bert_config)
        print(f"检测到模型类型: {model_type.upper()}")
        print(f"  - 隐藏层数: {bert_config.get('num_hidden_layers', 12)}")
        print(f"  - 隐藏层大小: {bert_config.get('hidden_size', 768)}")
        
    except Exception as e:
        print(f"✗ 读取配置文件失败: {e}")
        return False
    
    # 方法1: 尝试使用 transformers 的转换工具
    pytorch_dump_output = os.path.join(model_dir, "pytorch_model.bin")
    
    print("\n[方法1] 尝试使用 transformers 内置转换工具...")
    try:
        # 对于新版 transformers，我们直接使用 Python API
        from transformers import BertConfig
        
        # 创建 HF 配置
        hf_config = create_hf_config(bert_config, model_type)
        config_json_path = os.path.join(model_dir, "config.json")
        
        with open(config_json_path, 'w', encoding='utf-8') as f:
            json.dump(hf_config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 已创建 Hugging Face 配置文件: {config_json_path}")
        
        # 尝试使用 transformers 的转换函数
        if model_type == "bert":
            try:
                from transformers.models.bert.convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
                print("正在转换权重...")
                convert_tf_checkpoint_to_pytorch(
                    bert_config_file=bert_config_file,
                    tf_checkpoint_path=tf_checkpoint,
                    pytorch_dump_path=pytorch_dump_output
                )
                print(f"✓ 使用权重转换工具成功")
                
            except ImportError:
                print("⚠ transformers 中未找到转换模块，尝试其他方法...")
                raise ImportError("转换模块不存在")
        else:
            # RoBERTa 使用相同的转换逻辑
            try:
                from transformers.models.roberta.convert_roberta_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
                print("正在转换权重...")
                convert_tf_checkpoint_to_pytorch(
                    bert_config_file=bert_config_file,
                    tf_checkpoint_path=tf_checkpoint,
                    pytorch_dump_path=pytorch_dump_output
                )
                print(f"✓ 使用权重转换工具成功")
                
            except ImportError:
                print("⚠ RoBERTa 转换模块不存在，尝试通用方法...")
                raise ImportError("RoBERTa转换模块不存在")
        
        # 检查转换结果
        if os.path.exists(pytorch_dump_output):
            file_size = os.path.getsize(pytorch_dump_output) / (1024 * 1024)
            print(f"✓ 模型转换成功! PyTorch 模型大小: {file_size:.2f} MB")
            print(f"\n现在可以使用以下代码加载模型:")
            if model_type == "roberta":
                print(f'  from transformers import RobertaTokenizer, RobertaModel')
                print(f'  tokenizer = RobertaTokenizer.from_pretrained("{model_dir}")')
                print(f'  model = RobertaModel.from_pretrained("{model_dir}")')
            else:
                print(f'  from transformers import BertTokenizer, BertModel')
                print(f'  tokenizer = BertTokenizer.from_pretrained("{model_dir}")')
                print(f'  model = BertModel.from_pretrained("{model_dir}")')
            return True
        else:
            print("✗ 转换后未生成 pytorch_model.bin 文件")
            
    except Exception as e:
        print(f"✗ 方法1失败: {e}")
    
    # 方法2: 提示用户从 Hugging Face 下载
    print("\n[方法2] 推荐方案：从 Hugging Face 直接下载 PyTorch 格式模型")
    print("="*70)
    
    if model_type == "roberta":
        print("\n对于 RoBERTa 模型，建议直接从 Hugging Face 下载：")
        print("  pip install huggingface_hub")
        print(f"  hf download hfl/chinese-roberta-wwm-ext-large --local-dir {model_dir}")
        print("\n或者访问: https://huggingface.co/hfl/chinese-roberta-wwm-ext-large")
    else:
        print("\n对于 BERT 模型，建议直接从 Hugging Face 下载：")
        print("  pip install huggingface_hub")
        print(f"  hf download hfl/chinese-bert-wwm-ext --local-dir {model_dir}")
        print("\n或者访问: https://huggingface.co/hfl/chinese-bert-wwm-ext")
    
    print("\n如果已经手动下载了 pytorch_model.bin，请将其放入模型目录")
    print("="*70)
    
    # 最后检查
    if os.path.exists(pytorch_dump_output):
        print(f"\n✓ 发现已有的 PyTorch 模型文件，可以继续使用！")
        return True
    else:
        print(f"\n✗ 模型 '{model_name}' 转换失败")
        return False


def main():
    """主函数：批量转换所有模型"""
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(project_dir, "model")
    
    print("="*70)
    print("BERT/RoBERTa 模型批量转换工具")
    print("="*70)
    print(f"\n模型目录: {model_dir}\n")
    
    # 检查模型目录是否存在
    if not os.path.exists(model_dir):
        print(f"✗ 错误: 模型目录不存在 - {model_dir}")
        sys.exit(1)
    
    # 扫描所有子目录
    model_dirs = []
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path):
            # 检查是否有 TensorFlow 文件
            bert_config = os.path.join(item_path, "bert_config.json")
            if os.path.exists(bert_config):
                model_dirs.append((item, item_path))
    
    if not model_dirs:
        print("未找到需要转换的模型")
        return
    
    print(f"找到 {len(model_dirs)} 个模型目录:\n")
    for name, path in model_dirs:
        converted = check_model_converted(path)
        status = "✓ 已转换" if converted else "○ 待转换"
        print(f"  {status}: {name}")
    
    # 开始转换
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    print(f"\n{'='*70}")
    print("开始转换...")
    print(f"{'='*70}")
    
    for name, path in model_dirs:
        if check_model_converted(path):
            skip_count += 1
            continue
        
        if convert_single_model(path, name):
            success_count += 1
        else:
            fail_count += 1
    
    # 输出总结
    print(f"\n{'='*70}")
    print("转换完成统计")
    print(f"{'='*70}")
    print(f"总模型数: {len(model_dirs)}")
    print(f"已跳过（已完成）: {skip_count}")
    print(f"转换成功: {success_count}")
    print(f"转换失败: {fail_count}")
    
    if fail_count > 0:
        print(f"\n⚠ 有 {fail_count} 个模型转换失败，请参考上述提示手动处理")
        sys.exit(1)
    else:
        print("\n✓ 所有模型处理完成！")


if __name__ == "__main__":
    main()
