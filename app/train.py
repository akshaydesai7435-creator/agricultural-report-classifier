import argparse
import sys
from pathlib import Path
from app.core import milestone_3, DEFAULT_MODEL_DIR, DEFAULT_DATASET_PATH


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="agricultural-report-trainer",
        description="Fine-tune GPT-2 on agricultural reports dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the dataset JSON file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Directory to save the fine-tuned model and tokenizer."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per device."
    )
    args = parser.parse_args(argv)

    print(f"Starting training for {args.epochs} epoch(s)...")
    saved_path = milestone_3(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_path=args.dataset
    )
    print(f"Model and tokenizer successfully saved to: {saved_path}")


if __name__ == "__main__":
    main()

