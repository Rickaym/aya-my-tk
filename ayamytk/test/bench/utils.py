import csv
import pandas as pd
from collections import defaultdict

def grep_answer(answer, expected_answer):
    return expected_answer in answer


def load_csv_as_dicts(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

def reformat_mmlu_to_csv(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

def reformat_mmlu_to_single_row(file_path, output_path):
    # Read the CSV file
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader]

    # Group rows by sample_id
    grouped_data = defaultdict(dict)
    for row in rows:
        sample_id = row['sample_id']
        field = row['field']
        value = row['my_google']

        if field == 'question':
            grouped_data[sample_id]['question'] = value
        elif field == 'option_a':
            grouped_data[sample_id]['A'] = value
        elif field == 'option_b':
            grouped_data[sample_id]['B'] = value
        elif field == 'option_c':
            grouped_data[sample_id]['C'] = value
        elif field == 'option_d':
            grouped_data[sample_id]['D'] = value

    # Convert to list of dictionaries for writing to CSV
    reformatted_data = []
    for sample_id, data in grouped_data.items():
        row_data = {'sample_id': sample_id}
        row_data.update(data)
        reformatted_data.append(row_data)

    # Write to a new CSV file
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        fields = ['sample_id', 'question', 'A', 'B', 'C', 'D']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reformatted_data)

    print(f"Reformatted data saved to {output_path}")
    return reformatted_data


def add_answer_row(file_path, output_path):
    import pandas as pd
    from datasets import load_dataset

    # Load the dataset and CSV file
    gmmlu_lite = load_dataset("CohereForAI/Global-MMLU-Lite", 'en')
    df = pd.read_csv(file_path)

    # Print the first 10 sample IDs from the CSV file
    print("First 10 sample IDs from both sources:")
    for i, (csv_id, dataset_id) in enumerate(zip(df["sample_id"][:10], gmmlu_lite['test']['sample_id'][:10])):
        print(f"{i+1}. CSV: {csv_id} | Dataset: {dataset_id}")

    # Add answers to the dataframe
    df['answer'] = gmmlu_lite['test']['answer']

    # Verify sample IDs match
    assert all(df["sample_id"].values == gmmlu_lite['test']['sample_id']), "Sample IDs don't match"

    # Save to output file
    df.to_csv(output_path, index=False)
    print(f"Added answers and saved to {output_path}")

if __name__ == "__main__":
    # reformat_mmlu_to_single_row("./mmlu/mmlu_lite_my_400.csv", "./mmlu/mmlu_lite_my_400_single_row.csv")
    # add_answer_row("./mmlu/mmlu_lite_my_400_single_row.csv", "./mmlu/mmlu_lite_my_400_single_row_2.csv")
    import pandas as pd

    df = pd.read_csv("./mmlu/mmlu_lite_my_400_single_row.csv")

    # Split the sample_id field into subject and id
    df['subject'] = df['sample_id'].apply(lambda x: x.split('/')[0])

    # Save the updated dataframe
    df.to_csv("./mmlu/mmlu_lite_my_400_single_row_with_subject.csv", index=False)
    print("Added subject and id fields and saved to ./mmlu/mmlu_lite_my_400_single_row_with_subject.csv")
