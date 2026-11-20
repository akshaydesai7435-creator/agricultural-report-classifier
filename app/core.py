import json
from pathlib import Path
from typing import Any
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

CATEGORIES = [
    "PEST_ALERT",
    "SOIL_CONDITION",
    "WATER_ISSUE",
    "CROP_HEALTH",
]

DEFAULT_DATASET_PATH = Path("data/reports_dataset.json")
DEFAULT_MODEL_DIR = Path("saved_model")


def format_prompt(report: str) -> str:
    cleaned = report.strip()
    return f"Report: {cleaned}\nCategory:"


def format_example(report: str, label: str) -> str:
    return f"{format_prompt(report)} {label}"


def load_dataset(path: str | Path | None = None) -> list[dict[str, str]]:
    dataset_file = Path(path or DEFAULT_DATASET_PATH)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    with dataset_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of records.")
    return data


def milestone_1(path: str | Path | None = None) -> list[dict[str, str]]:
    data = load_dataset(path)
    for item in data:
        if "text" not in item or "label" not in item:
            raise ValueError("Each record must contain 'text' and 'label'.")
        if item["label"] not in CATEGORIES:
            raise ValueError(f"Invalid label: {item['label']}")
    return data


def get_tokenizer(model_name_or_path: str = "gpt2") -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": CATEGORIES})
    return tokenizer


def get_model(tokenizer: GPT2Tokenizer, model_name_or_path: str = "gpt2") -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    return model


def milestone_2(model_name_or_path: str = "gpt2") -> tuple[GPT2Tokenizer, GPT2LMHeadModel]:
    tokenizer = get_tokenizer(model_name_or_path)
    model = get_model(tokenizer, model_name_or_path)
    return tokenizer, model


def prepare_dataset_for_training(tokenizer: GPT2Tokenizer, raw_data: list[dict[str, str]], max_length: int = 64):
    texts = [f"{format_example(item['text'], item['label'])}{tokenizer.eos_token}" for item in raw_data]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels = input_ids.clone()

    prompt_texts = [f"{format_prompt(item['text'])} " for item in raw_data]
    prompt_encodings = tokenizer(prompt_texts, truncation=True, max_length=max_length)

    for i, p_ids in enumerate(prompt_encodings["input_ids"]):
        prompt_len = len(p_ids)
        labels[i, :prompt_len] = -100

    labels[attention_mask == 0] = -100

    class AgriculturalDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

    return AgriculturalDataset(encodings, labels)


def milestone_3(
    output_dir: str | Path = DEFAULT_MODEL_DIR,
    epochs: int = 5,
    batch_size: int = 4,
    dataset_path: str | Path | None = None
) -> str:
    from transformers import Trainer, TrainingArguments

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_data = load_dataset(dataset_path)
    tokenizer, model = milestone_2("gpt2")
    train_dataset = prepare_dataset_for_training(tokenizer, raw_data)

    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=1e-3,
        logging_steps=10,
        save_strategy="no",
        use_cpu=not torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    trainer.train()

    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    return str(output_path)


def load_classifier(model_dir: str | Path = DEFAULT_MODEL_DIR) -> tuple[GPT2Tokenizer, GPT2LMHeadModel]:
    path = Path(model_dir)
    if not path.exists() or not (path / "config.json").exists():
        raise FileNotFoundError(f"Model directory or weights not found at: {path}")
    tokenizer = GPT2Tokenizer.from_pretrained(str(path))
    model = GPT2LMHeadModel.from_pretrained(str(path))
    model.eval()
    return tokenizer, model


def predict_category(
    report: str,
    tokenizer: GPT2Tokenizer | None = None,
    model: GPT2LMHeadModel | None = None,
    model_dir: str | Path = DEFAULT_MODEL_DIR
) -> str:
    cleaned = report.strip() if report else ""
    if not cleaned:
        raise ValueError("Report cannot be empty.")

    if tokenizer is None or model is None:
        tokenizer, model = load_classifier(model_dir)

    prompt = f"{format_prompt(cleaned)} "
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    generated_ids = output[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    for cat in CATEGORIES:
        if cat in generated_text:
            return cat

    tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    for tok in tokens:
        if tok in CATEGORIES:
            return tok

    category_token_ids = [tokenizer.convert_tokens_to_ids(cat) for cat in CATEGORIES]
    logits = model(**inputs).logits[:, -1, :]
    category_logits = logits[0, category_token_ids]
    best_idx = int(torch.argmax(category_logits).item())
    return CATEGORIES[best_idx]


def milestone_4(report: str, model_dir: str | Path = DEFAULT_MODEL_DIR) -> str:
    return predict_category(report, model_dir=model_dir)


def milestone_5(file_path: str | Path, model_dir: str | Path = DEFAULT_MODEL_DIR) -> list[tuple[str, str]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    reports = [line for line in lines if line]
    if not reports:
        return []

    tokenizer, model = load_classifier(model_dir)
    results = []
    for rep in reports:
        pred = predict_category(rep, tokenizer=tokenizer, model=model)
        results.append((rep, pred))
    return results
