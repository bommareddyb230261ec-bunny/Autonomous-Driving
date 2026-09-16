import argparse
import json
from pathlib import Path

METRIC_COLUMNS = ("ade1s", "ade2s", "ade3s", "avgade", "failure_rate")


def calculate_averages(input_path: Path) -> tuple[int, dict[str, float]]:
    totals = {column: 0.0 for column in METRIC_COLUMNS}
    row_count = 0

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc

            missing_columns = [column for column in METRIC_COLUMNS if column not in row]
            if missing_columns:
                raise ValueError(
                    f"Missing columns on line {line_number}: {', '.join(missing_columns)}"
                )

            try:
                for column in METRIC_COLUMNS:
                    totals[column] += float(row[column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-numeric metric on line {line_number}") from exc

            row_count += 1

    if row_count == 0:
        raise ValueError("The input JSONL file contains no data rows.")

    return row_count, {column: total / row_count for column, total in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate column-wise averages from an OpenEMMA ade_results.jsonl file."
    )
    parser.add_argument("input_file", type=Path, help="Path to ade_results.jsonl")
    args = parser.parse_args()

    if not args.input_file.is_file():
        parser.error(f"Input file does not exist: {args.input_file}")

    try:
        row_count, averages = calculate_averages(args.input_file)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Rows processed: {row_count}")
    print("Overall column averages:")
    for column, average in averages.items():
        print(f"{column}: {average:.6f}")


if __name__ == "__main__":
    main()
