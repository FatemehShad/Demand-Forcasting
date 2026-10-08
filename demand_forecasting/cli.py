"""Command line training, temporal evaluation, and inference."""
import argparse
import json
from pathlib import Path

from .model import DemandForecaster, evaluate, read_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--train", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--horizon", type=int, default=12)
    train.add_argument("--iterations", type=int, default=180)
    ev = commands.add_parser("evaluate")
    ev.add_argument("--train", required=True)
    ev.add_argument("--output-dir", default="artifacts")
    ev.add_argument("--horizon", type=int, default=12)
    ev.add_argument("--iterations", type=int, default=180)
    pred = commands.add_parser("predict")
    pred.add_argument("--model", required=True)
    pred.add_argument("--test", required=True)
    pred.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "train":
        model = DemandForecaster(args.horizon, args.iterations).fit(read_data(args.train, training=True))
        model.save(args.model)
        print(f"Saved model to {args.model}; history cutoff {model.cutoff_.date()}")
    elif args.command == "evaluate":
        report, rows = evaluate(read_data(args.train, training=True),
                                horizon=args.horizon, iterations=args.iterations)
        folder = Path(args.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        rows.to_csv(folder / "validation_predictions.csv", index=False)
        print(json.dumps(report, indent=2))
    else:
        data = read_data(args.test)
        model = DemandForecaster.load(args.model)
        result = data[["record_ID"]].copy()
        result["units_sold"] = model.predict(data)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Saved {len(result)} forecasts to {args.output}")


if __name__ == "__main__":
    main()
