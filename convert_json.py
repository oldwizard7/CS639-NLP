import pandas as pd
import json
import os

# File paths
input_file = os.path.expanduser("~/Desktop/generations_math500_sft.jsonl")
output_file = os.path.expanduser("~/Desktop/reasoning_units_math500_sft.jsonl")

# Read original jsonl file
df = pd.read_json(input_file, lines=True)

def split_into_reasoning_units(text):
    """
    Split raw model output into reasoning units using blank lines,
    but avoid splitting inside markdown code blocks.
    """
    if not isinstance(text, str):
        return []

    units = []
    current_unit = []
    inside_code_block = False

    lines = text.splitlines()

    for line in lines:
        stripped = line.strip()

        # Detect start/end of markdown code block
        if stripped.startswith("```"):
            inside_code_block = not inside_code_block
            current_unit.append(line)
            continue

        # Blank line outside code block means new reasoning unit
        if stripped == "" and not inside_code_block:
            if current_unit:
                unit = "\n".join(current_unit).strip()
                if unit:
                    units.append(unit)
                current_unit = []
        else:
            current_unit.append(line)

    # Add final unit
    if current_unit:
        unit = "\n".join(current_unit).strip()
        if unit:
            units.append(unit)

    return units


# Process every row
processed_rows = []

for idx, row in df.iterrows():
    reasoning_units = split_into_reasoning_units(row["raw_output"])

    new_row = {
        "problem_id": row["problem_id"],
        "model_tag": row["model_tag"],
        "model_name": row["model_name"],
        "sample_id": int(row["sample_id"]),
        "finish_reason": row["finish_reason"],
        "reasoning_units": reasoning_units,
        "num_reasoning_units": len(reasoning_units)
    }

    processed_rows.append(new_row)


# Save to jsonl
with open(output_file, "w", encoding="utf-8") as f:
    for row in processed_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Saved reasoning units to: {output_file}")
print(f"Total rows saved: {len(processed_rows)}")

# Check one example
print("\nExample row:")
print(processed_rows[0])