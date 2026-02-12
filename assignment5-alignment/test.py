from transformers import AutoTokenizer
from pathlib import Path
import inspect
import torch

# 使用本地已有的 tokenizer（不需要网络连接）
fixtures_path = Path(__file__).parent / "tests" / "fixtures"
tokenizer_path = fixtures_path / "Meta-Llama-3-8B"

# 加载本地 tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# 重要：设置 pad_token（Llama tokenizer 默认没有）
# 通常使用 eos_token 作为 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 准备一批文本
sentences = [
    "I love using Transformers.",
    "unbelievable"  # 这句话比上一句短
]

# 3. 使用 __call__ 方法（推荐方式）
batch_inputs = tokenizer(
    sentences,
    padding=True,          # 填充到当前批次最长句子的长度
    truncation=True,       # 如果超过模型最大长度则截断
    max_length=10,         # 限制最大长度
    return_tensors="pt"    # 返回 PyTorch 张量
)
print("batch_inputs:")
print(batch_inputs)
print("Input IDs (数字编码):")
print(batch_inputs["input_ids"]) 

print("\nAttention Mask (告诉模型不看Padding部分):")
print(batch_inputs["attention_mask"])

# 4. 解码 (ID -> 文本)
decoded_text = tokenizer.decode(batch_inputs["input_ids"][0], skip_special_tokens=False)
print(f"\n解码结果: {decoded_text}")

text = tokenizer.decode([0,1,2,3,4,5,6,7,8,9,10])
print(f"\n解码结果: {text}")

tokens = tokenizer.tokenize(sentences[1])
print(f"\n分词结果: {tokens}")

prompt_tokenized = tokenizer(
        ["I love using Transformers.","unbelievable","It is fast."],
        padding=False,
        truncation=False,
        return_tensors=None,  # 返回列表而不是tensor
        add_special_tokens=True
    )
print(prompt_tokenized)