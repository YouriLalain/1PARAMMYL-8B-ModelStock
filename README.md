---
base_model:
- arcee-ai/Llama-3.1-SuperNova-Lite
tags:
- merge
- mergekit
- lazymergekit
- arcee-ai/Llama-3.1-SuperNova-Lite
---

# 1PARAMMYL-8B-ModelStock

1PARAMMYL-8B-ModelStock is a merge of the following models using [LazyMergekit](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing):
* [arcee-ai/Llama-3.1-SuperNova-Lite](https://huggingface.co/arcee-ai/Llama-3.1-SuperNova-Lite)

## 🧩 Configuration

```yaml
slices:
  - sources:
      - model: arcee-ai/Llama-3.1-SuperNova-Lite
        layer_range: [0, 32]
      - model: DreadPoor/Heart_Stolen-8B-Model_Stock
        layer_range: [0, 32]
      - model: Dampfinchen/Llama-3.1-8B-Ultra-Instruct
        layer_range: [0, 32]
merge_method: model_stock
base_model: arcee-ai/Llama-3.1-SuperNova-Lite
dtype: bfloat16
```

## 💻 Usage

```python
!pip install -qU transformers accelerate

from transformers import AutoTokenizer
import transformers
import torch

model = "Youlln/1PARAMMYL-8B-ModelStock"
messages = [{"role": "user", "content": "What is a large language model?"}]

tokenizer = AutoTokenizer.from_pretrained(model)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
```