from pathlib import Path
import pytest
from app.core import (
    CATEGORIES,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    format_prompt,
    format_example,
    predict_category,
)


def test_milestone_1(tmp_path):
    data = milestone_1()
    assert isinstance(data, list)
    assert len(data) >= 16

    labels = {item["label"] for item in data}
    assert labels == set(CATEGORIES)
    for cat in CATEGORIES:
        count = sum(1 for item in data if item["label"] == cat)
        assert count >= 4


def test_milestone_2():
    tokenizer, model = milestone_2("gpt2")

    for cat in CATEGORIES:
        token_id = tokenizer.convert_tokens_to_ids(cat)
        assert token_id != tokenizer.unk_token_id
        assert cat in tokenizer.all_special_tokens

    assert model.get_input_embeddings().weight.shape[0] == len(tokenizer)


def test_prompt_formatting():
    prompt = format_prompt("Wheat leaves have yellow rust.")
    assert prompt == "Report: Wheat leaves have yellow rust.\nCategory:"
    example = format_example("Wheat leaves have yellow rust.", "CROP_HEALTH")
    assert example == "Report: Wheat leaves have yellow rust.\nCategory: CROP_HEALTH"


def test_milestone_4_and_5_file_handling(tmp_path):
    with pytest.raises(FileNotFoundError):
        milestone_4("Test report", model_dir=tmp_path / "nonexistent")

    with pytest.raises(ValueError):
        milestone_4("   ", model_dir=tmp_path / "nonexistent")

    with pytest.raises(FileNotFoundError):
        milestone_5(tmp_path / "missing.txt", model_dir=tmp_path / "nonexistent")
