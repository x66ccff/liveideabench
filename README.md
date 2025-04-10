# 🤖💡 [LiveIdeaBench](http://liveideabench.com): Evaluating LLMs' Scientific Creativity and Idea Generation with Minimal Context


_"It's not like finding a needle in a haystack, it is like creating new needles."_


🏆 Leaderboard: http://liveideabench.com 💡

### Dataset

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DatasetV1-yellow)](https://huggingface.co/datasets/6cf/liveideabench)

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DatasetV2-yellow)](https://huggingface.co/datasets/6cf/liveideabench-v2)

### Paper

[![arXiv](https://img.shields.io/badge/arXiv-2412.17596-b31b1b.svg)](https://arxiv.org/abs/2412.17596)


### 🧠✨🎉 News (2025/3/29): Latest Dataset Update (v2) on Hugging Face! 

We are pleased to announce that, based on the invaluable feedback from reviewers, we have enhanced our benchmark by upgrading it to **version 2**. This update introduces a new dimension—**Clarity**—and improves the prompts, evaluation process (including the rejection handling mechanism), making our benchmark more comprehensive and objective.

This v2 version of the benchmark incorporates the latest models, including: `claude-3.7-sonnet:thinking`, `o3-mini-high`, `gpt-4.5-preview`, `qwq-32b`, `deepseek-r1`, `gemini-2.0-flash-thinking`, and a total of **41** state-of-the-art models.

### 🧠✨🎉 News (2025/1/27): Latest Dataset Update on Hugging Face! 

We are excited to announce that the latest dataset, including supplementary tests for models like **deepseek-R1**, **deepseek-V3**, **minimax-01**, **phi-4**, and **Opus**, has been uploaded to Hugging Face! 🚀

Check it out here: [https://huggingface.co/datasets/6cf/liveideabench-DLC-250127](https://huggingface.co/datasets/6cf/liveideabench-DLC-250127)

---

## LiveIdeaBench Evaluation Framework
![LiveIdeaBench Evaluation Framework](./assets/image.png)
![Leaderboard](./assets/bench.png)

## Evaluation Instruction

### 1️⃣ Database Initialization

Run the Python script to initialize the database:
```bash
python -c "from utils.database import init_database; init_database()"
```

### 2️⃣ Configuring API Keys

Before running the program, you need to configure at least one API key:

1. Create an `apikey` file and write your OpenRouter API key:
   ```bash
   echo "your-openrouter-api-key" > apikey
   ```

   Alternatively, set environment variables:
   ```bash
   export OPENROUTER_API_KEY="your-openrouter-api-key"
   export STEP_API_KEY="your-step-api-key"
   export GEMINI_API_KEYS="key1,key2,key3"
   ```

### 3️⃣ Running Examples

Generate and evaluate ideas using a specified model:

```bash
# Generate ideas using a specified model
python run.py --idea_model "openai/gpt-4o-mini"

# Use a specific provider
python run.py --idea_model "openai/gpt-4o-mini" --provider openrouter
```

```bash
# Use a single keyword:

python run.py --idea_model "openai/gpt-4o-mini" --keyword "relativity"
# Use multiple keywords:

python run.py --idea_model "openai/gpt-4o-mini" --keyword "relativity" "periodic table"
# Do not specify a keyword (use all keywords):

python run.py --idea_model "openai/gpt-4o-mini"
```

### 4️⃣ Database Export

```bash
python view_database.py      
```
Then, run `stats.ipynb`, to generate `data/data.parquet`

### 5️⃣ Evaluate Fluency

```bash
python hash.py
```

### 6️⃣ Compute Flexibility & Plotting

run `stats_flexibility.ipynb`


## 🌍🌱 CO2 Emission Estimation

run `co2.ipynb`




## Bibtex


```bibtex
@article{ruan2024liveideabench,
title={LiveIdeaBench: Evaluating LLMs' Scientific Creativity and Idea Generation with Minimal Context},
author={Kai Ruan and Xuan Wang and Jixiang Hong and Peng Wang and Yang Liu and Hao Sun},
journal={arXiv preprint arXiv:2412.17596},
year={2024}
}
```
