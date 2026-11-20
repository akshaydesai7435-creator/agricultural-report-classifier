import argparse
import sys
from pathlib import Path
from app.core import milestone_4, milestone_5, DEFAULT_MODEL_DIR


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="agricultural-report-classifier",
        description="Agronomists use this tool to automatically classify short farmer reports or field observations to monitor farm health and respond proactively."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    predict_parser = sub.add_parser("predict", help="classify a single report string")
    predict_parser.add_argument("--report", type=str, required=True, help="report text to classify")
    predict_parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR), help="path to model directory")

    file_parser = sub.add_parser("predict-file", help="classify reports from a text file")
    file_parser.add_argument("--file", type=str, required=True, help="path to text file containing reports")
    file_parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR), help="path to model directory")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Agricultural Report Classifier is ready. Open README.md for your milestones.")
    elif args.command == "predict":
        try:
            category = milestone_4(args.report, model_dir=args.model_dir)
            print(f"Report: {args.report}")
            print(f"Predicted Category: {category}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except FileNotFoundError as e:
            print(f"Error: Model not found at '{args.model_dir}'. Run training first or check path.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "predict-file":
        try:
            results = milestone_5(args.file, model_dir=args.model_dir)
            if not results:
                print(f"No non-empty reports found in {args.file}")
                return
            for report, category in results:
                print(f"Report: {report}")
                print(f"Predicted Category: {category}")
                print("-" * 40)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
