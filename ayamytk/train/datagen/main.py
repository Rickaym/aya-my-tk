from generate import Generator
import pandas as pd
import json
import templates

# header: alphabet,word,phonetics,meaning,pos,origin
df = pd.read_csv("Burmese-Dictionary/burmese_dictionary.csv")

# Ensure string type for relevant columns and handle potential NaN values
df["word"] = df["word"].astype(str).fillna("")
df["meaning"] = df["meaning"].astype(str).fillna("")
df["phonetics"] = df["phonetics"].astype(str).fillna("")
df["origin"] = df["origin"].astype(str).fillna("")
# Add handling for the new 'alphabet' column
df["alphabet"] = df["alphabet"].astype(str).fillna("")

generator = Generator(
    [
        [
            templates.word_meaning_formal,
            templates.word_meaning_casual,
            templates.word_meaning_question,
            templates.word_meaning_polite,
            templates.word_meaning_rude,
            templates.word_meaning_academic,
        ],
        [
            templates.reverse_lookup_formal,
            templates.reverse_lookup_casual,
            templates.reverse_lookup_question,
            templates.reverse_lookup_academic,
        ],
        [
            templates.phonetic_formal,
            templates.phonetic_casual,
            templates.phonetic_question,
            templates.phonetic_academic,
        ],
        [
            templates.etymology_formal,
            templates.etymology_casual,
            templates.etymology_question,
            templates.etymology_academic,
        ],
        [
            templates.pos_formal,
            templates.pos_casual,
            templates.pos_question,
            templates.pos_academic,
        ],
    ],
    df,
)
data = generator.generate()

print(f"Finished generating")

# with open("finetuning_data_2.jsonl", "w", encoding="utf-8") as f:
# f.writelines(data)

with open("finetuning_data_cleaned_2.jsonl", "w", encoding="utf-8") as f:
    for obj in data:
        line = json.dumps(obj, ensure_ascii=False)
        f.write(line.strip() + "\n")
