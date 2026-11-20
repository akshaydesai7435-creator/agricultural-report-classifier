# Agricultural Report Classifier

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/mtgvjrw62f/badge.svg)](https://ask.thili.ai/b/mtgvjrw62f)

This project develops an Agricultural Report Classifier using a fine-tuned generative GPT-2 model. It helps agronomists automatically categorize farmer reports into four specific operational categories to improve farm monitoring and response.

## Categories

The classifier categorizes reports into exactly four categories registered as special tokens:
- `PEST_ALERT`
- `SOIL_CONDITION`
- `WATER_ISSUE`
- `CROP_HEALTH`

## Dataset

The dataset is stored in `data/reports_dataset.json` as a balanced JSON collection containing realistic field observations mapped to each of the four categories.
A sample multi-line input file is also provided at `data/sample_reports.txt`.

## Installation

Create a virtual environment and install the required dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Training

Train or fine-tune GPT-2 on the agricultural dataset using Hugging Face `Trainer`. The script resizes the model's token embeddings for the custom special tokens and trains the model to generate the category label token after the report prompt. Training automatically uses CPU when CUDA is unavailable.

To train the model:

```powershell
python -m app.train --epochs 3 --batch-size 8
```

Options:
- `--dataset`: Path to dataset JSON file (default: `data/reports_dataset.json`)
- `--output-dir`: Output directory to save the fine-tuned model and tokenizer (default: `saved_model`)
- `--epochs`: Number of training epochs (default: `3`)
- `--batch-size`: Batch size per device (default: `4`)

## CLI Usage

### Check plumbing

```powershell
python -m app.cli hello
```

### Classify a single report

```powershell
python -m app.cli predict --report "Leaves have yellow spots and insects are damaging the crop"
```

### Classify reports from a file

```powershell
python -m app.cli predict-file --file data/sample_reports.txt
```

## Running Tests

Run the test suite using pytest:

```powershell
python -m pytest -q
```
