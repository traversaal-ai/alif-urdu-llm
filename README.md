# Alif: Advancing Urdu Large Language Models via Multilingual Synthetic Data Distillation


- Paper: In-progress
- [Datasets](https://huggingface.co/datasets/large-traversaal/urdu-instruct) (UrduInstruct and Urdu Eval)
- [Model](https://huggingface.co/large-traversaal/Alif-1.0-8B-Instruct)
- [Blog](https://blog.traversaal.ai/announcing-alif-1-0-our-first-urdu-llm-outperforming-other-open-source-llms/)
- [Live Demo](https://huggingface.co/spaces/large-traversaal/Alif-1.0-8B-Instruct)

## Abstract:

Developing a high-performing large language models (LLMs) for low-resource languages such as Urdu, present several challenges. These challenges include the scarcity of high-quality datasets, multilingual inconsistencies, and safety concerns. Existing multilingual LLMs often address these issues by translating large volumes of available data. However, such translations often lack quality and cultural nuance while also incurring significant costs for data curation and training. To address these issues, we propose Alif-1.0-8B-Instruct, a multilingual Urdu-English model, that tackles these challenges with a unique approach. We train the model on a high-quality, multilingual synthetic dataset (Urdu-Instruct), developed using a modified self-instruct technique. By using unique prompts and seed values for each task along with a global task pool, this dataset incorporates Urdu-native chain-of-thought based reasoning, bilingual translation, cultural relevance, and ethical safety alignments. This technique significantly enhances the comprehension of Alif-1.0-8B-Instruct model for Urdu-specific tasks. As a result, Alif-1.0-8B-Instruct, built upon the pretrained Llama-3.1-8B, demonstrates superior performance compared to Llama-3.1-8B-Instruct for Urdu specific-tasks. It also outperformed leading multilingual LLMs, including Mistral-7B-Instruct-v0.3, Qwen-2.5-7B-Instruct, and Cohere-Aya-Expanse-8B, all within a training budget of under \$100. Our results demonstrate that high-performance and low-resource language LLMs can be developed efficiently and culturally aligned using our modified self-instruct approach.

## Citation

If you find Curator Evals useful, do not forget to cite us!

```
In-progress
```

