import csv

def grep_answer(answer, expected_answer):
    return expected_answer in answer


def load_csv_as_dicts(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [row for row in reader]
