# Aya Myanmar Toolkit
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

This toolkit is developed for evaluating language models, and synthetic data generation for the Burmese language. This repository contains a data generation pipeline, and evaluation scripts for various language models on Burmese language tasks.

## 🎯 Benchmarks

### Primary Benchmarks

| Benchmark Name | Questions | Types |
|--------------|------|------------|
| [Myanmar G12L Benchmark](https://huggingface.co/datasets/Rickaym/Myanmar-G12L-Benchmark) | 962 | multiple-choice, true-false, short-answer, fill-in-the-blank, long-answer |
| [MMLU Lite](https://huggingface.co/datasets/Rickaym/Burmese-MMLU-Lite) | 600 | multiple-choice  |
| Driving and Riding Theory Test (Myanmar) | 200 | multiple-choice  |
| [Belebele](https://huggingface.co/datasets/facebook/belebele/viewer/mya_Mymr?row=0&views%5B%5D=mya_mymr) | 900 | multiple-choice |


## 📊 Model Performance

Coming soon

## 🗃️ Training Datasets

| Dataset Name | Rows | Tokens |
|--------------|------|------------|
| [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | 109k | 24B |
| Myanmar Instructions 67k | 67k | - |
| [Alpaca Burmese Cleaned](https://huggingface.co/datasets/saillab/alpaca-myanmar_burmese-cleaned) | 52k | 2M |
| [OSCAR-2019-Burmese-fix](https://huggingface.co/datasets/5w4n/OSCAR-2019-Burmese-fix) | 140k | - |
| [Myanmar Dictionary](https://huggingface.co/datasets/Rickaym/Burmese-Dictionary) | - | 220k |
| Public School Text Books | - | 680k |

### Additional Datasets

| Benchmark Name | Status |
|--------------|--------|
| OSCAR-2301 | Coming Soon |
| myParaphrase | Coming Soon |
| myanmar_news | Coming Soon |
| FLORES-200 | Coming Soon |
| myPOS | Coming Soon |
| BurmeseProverbDataSet | Coming Soon |
| Wattpad | Coming Soon |


## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- uv

### Installation

```bash
# Clone the repository
git clone https://github.com/Rickaym/aya-my-tk.git
cd aya-my-tk

# Install dependencies
uv sync
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For any questions or suggestions, please open an issue in the repository.

---
*This project started as part of the Expedition Aya 2025 initiative.*
