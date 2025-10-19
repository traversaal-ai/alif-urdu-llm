<div align="center">

# Alif: Advancing Urdu Large Language Models via Multilingual Synthetic Data Distillation

<br>

<p align="center">📄 
  <a href="https://arxiv.org/abs/2510.09051">[ Paper ]</a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/large-traversaal/urdu-instruct">[ Datasets ]</a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/large-traversaal/Alif-1.0-8B-Instruct">[ Model ]</a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://blog.traversaal.ai/announcing-alif-1-0-our-first-urdu-llm-outperforming-other-open-source-llms/">[ Blog ]</a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/spaces/large-traversaal/Alif-1.0-8B-Instruct">[ Live Demo ]</a>
</p>

<br>

</div>



<!--
# Alif: Advancing Urdu Large Language Models via Multilingual Synthetic Data Distillation


- [Paper](https://arxiv.org/abs/2510.09051)
- [Datasets](https://huggingface.co/datasets/large-traversaal/urdu-instruct) (UrduInstruct and UrduEval)
- [Model](https://huggingface.co/large-traversaal/Alif-1.0-8B-Instruct)
- [Blog](https://blog.traversaal.ai/announcing-alif-1-0-our-first-urdu-llm-outperforming-other-open-source-llms/)
- [Live Demo](https://huggingface.co/spaces/large-traversaal/Alif-1.0-8B-Instruct) -->

## Abstract:

Developing a high-performing large language models (LLMs) for low-resource languages such as Urdu, present several challenges. These challenges include the scarcity of high-quality datasets, multilingual inconsistencies, and safety concerns. Existing multilingual LLMs often address these issues by translating large volumes of available data. However, such translations often lack quality and cultural nuance while also incurring significant costs for data curation and training. To address these issues, we propose Alif-1.0-8B-Instruct, a multilingual Urdu-English model, that tackles these challenges with a unique approach. We train the model on a high-quality, multilingual synthetic dataset (Urdu-Instruct), developed using a modified self-instruct technique. By using unique prompts and seed values for each task along with a global task pool, this dataset incorporates Urdu-native chain-of-thought based reasoning, bilingual translation, cultural relevance, and ethical safety alignments. This technique significantly enhances the comprehension of Alif-1.0-8B-Instruct model for Urdu-specific tasks. As a result, Alif-1.0-8B-Instruct, built upon the pretrained Llama-3.1-8B, demonstrates superior performance compared to Llama-3.1-8B-Instruct for Urdu specific-tasks. It also outperformed leading multilingual LLMs, including Mistral-7B-Instruct-v0.3, Qwen-2.5-7B-Instruct, and Cohere-Aya-Expanse-8B, all within a training budget of under \$100. Our results demonstrate that high-performance and low-resource language LLMs can be developed efficiently and culturally aligned using our modified self-instruct approach.

## Urdu-Instruct (train) and Urdu-Eval (test)

The Urdu-Instruct dataset is a high-quality multilingual synthetic corpus containing 51,686 training and 1,084 test examples. It was generated using GPT-4o under a modified self-instruction framework to improve instruction-following, reasoning, and bilingual understanding in Urdu. This dataset is part of the Alif-1.0-8B-Instruct project and was created by the Traversaal.AI Research Team. It supports seven essential Urdu-language tasks: Generation, Reasoning, Ethics, Question Answering, Translation, Classification, and Sentiment Analysis.

<div align="center">

### Category-wise Distribution

| category         | train (Urdu_Instruct) | test (Urdu-Eval) |
|:-----------------|---------------:|--------------:|
| Translation      | 10,001         | 161           |
| Reasoning        | 9,590          | 170           |
| Ethics           | 9,002          | 156           |
| QA               | 8,177          | 149           |
| Generation       | 5,907          | 144           |
| Classification   | 4,662          | 152           |
| Sentiment        | 4,347          | 152           |
| **Total**        | **51,686**     | **1,084**     |

</div>


## Alif-1.0-8B-Instruct Model
Alif-1.0-8B-Instruct is a multilingual Urdu–English large language model built on Meta-Llama-3.1-8B. It was continued-pretrained on 200K Urdu Wikipedia articles and fine-tuned on 105K examples, including the Urdu-Instruct dataset. Using a modified self-instruct method, Alif improves Urdu reasoning, translation, and cultural understanding while retaining strong English fluency. Trained under $100 using LoRA adapters, Alif outperforms Llama-3.1-8B-Instruct, Qwen-2.5-7B, and Aya-Expanse-8B on Urdu benchmarks such as MGSM, AlpacaEval, and Dolly QA, achieving a 75.5 average score.

## Citation

If you find this repository useful, do not forget to cite us!

```bash
@article{ShafiqueAlif2025,
  title        = {Alif: Advancing Urdu Large Language Models via Multilingual Synthetic Data Distillation},
  author       = {Muhammad Ali Shafique and Kanwal Mehreen and Muhammad Arham and Maaz Amjad and Sabur Butt and Hamza Farooq},
  journal      = {arXiv preprint arXiv:2510.09051},
  year         = {2025},
  url          = {https://arxiv.org/abs/2510.09051}
}
```

