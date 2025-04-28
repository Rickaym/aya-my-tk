import pandas as pd
import json

class Templates:
    forward_lookup = [
        {
            "role": "System ",
            "content": "သင်သည် မြန်မာအဘိဓာန်ကို ထဲထဲဝင်ဝင် နားလည်သူဖြစ်ပြီ ဝေါဟာရအဓိပ္ပါယ်များနှင့်ပက်သက်၍ တိကျသေခြာစွာဖြေဆိုနိုင်သော လက်ထောက်တစ်ယောက် ဖြစ်သည်။",
        },
        {"role": "User", "content": "ဒီဝေါဟာရ \"{{word}}\" ရဲ့အဓိပ္ပါယ်က ဘာလဲ။"},
        {"role": "Chatbot", "content": "{{definition}}"},
    ]

    reverse_lookup = [
        {
            "role": "System ",
            "content": "သင်သည် မြန်မာအဘိဓာန်ကို ထဲထဲဝင်ဝင် နားလည်သူဖြစ်ပြီ ဝေါဟာရအဓိပ္ပါယ်များနှင့်ပက်သက်၍ တိကျသေခြာစွာဖြေဆိုနိုင်သော လက်ထောက်တစ်ယောက် ဖြစ်သည်။",
        },
        {"role": "User", "content": "ဒီအဓိပ္ပါယ် \"{{definition}}\" နဲ့သက်ဆိုင်တဲ့ ဝေါဟာရ ကိုပေးပါ။"},
        {"role": "Chatbot", "content": "{{word}}"},
    ]

    phonetic = [
        {
            "role": "System ",
            "content": "သင်သည် မြန်မာအဘိဓာန်ကို ထဲထဲဝင်ဝင် နားလည်သူဖြစ်ပြီ ဝေါဟာရအဓိပ္ပါယ်များနှင့်ပက်သက်၍ တိကျသေခြာစွာဖြေဆိုနိုင်သော လက်ထောက်တစ်ယောက် ဖြစ်သည်။",
        },
        {"role": "User", "content": "ဒီစာလုံး {{word}} ရဲ့အသံထွက်ကိုပေးပါ။" },
        {"role": "Chatbot", "content": "{{phonetics}}"},
    ]

    etymology = [
        {
            "role": "System ",
            "content": "သင်သည် မြန်မာအဘိဓာန်ကို ထဲထဲဝင်ဝင် နားလည်သူဖြစ်ပြီ ဝေါဟာရအဓိပ္ပါယ်များနှင့်ပက်သက်၍ တိကျသေခြာစွာဖြေဆိုနိုင်သော လက်ထောက်တစ်ယောက် ဖြစ်သည်။",
        },
        {"role": "User", "content": "ဒီစာလုံး {{word}} ရဲ့ ဇာစ်မြစ် ကိုပေးပါ။" },
        {"role": "Chatbot", "content": "{{word}} သည် {{origin}} မှဆင်းသက်လာခြင်း ဖြစ်သည်။" },
    ]

    alphabetical_indexing = [
         {
            "role": "System ",
            "content": "သင်သည် မြန်မာအဘိဓာန်ကို ထဲထဲဝင်ဝင် နားလည်သူဖြစ်ပြီ ဝေါဟာရအဓိပ္ပါယ်များနှင့်ပက်သက်၍ တိကျသေခြာစွာဖြေဆိုနိုင်သော လက်ထောက်တစ်ယောက် ဖြစ်သည်။",
        },
        {"role": "User", "content": "ဒီစာလုံး {{word}} သည် မြန်မာအက္ခရာထဲတွင် ဘယ်အက္ခရာအုပ်စု (က, ခ, ဂ, ...) မှာပါဝင်ပါသလဲ။"},
        {"role": "Chatbot", "content": "{{alphabet}}"},
    ]

# header: alphabet,word,phonetics,meaning,pos,origin
df = pd.read_csv("Burmese-Dictionary/burmese_dictionary.csv")

# Ensure string type for relevant columns and handle potential NaN values
df['word'] = df['word'].astype(str).fillna('')
df['meaning'] = df['meaning'].astype(str).fillna('')
df['phonetics'] = df['phonetics'].astype(str).fillna('')
df['origin'] = df['origin'].astype(str).fillna('')
# Add handling for the new 'alphabet' column
df['alphabet'] = df['alphabet'].astype(str).fillna('')

output_file = "finetuning_data.jsonl"

with open(output_file, 'w', encoding='utf-8') as f:
    for index, row in df.iterrows():
        word = row['word']
        definition = row['meaning']
        phonetics = row['phonetics']
        origin = row['origin']
        alphabet = row['alphabet'] # Extract the alphabet index

        # Skip rows with empty essential fields
        if not word:
            continue

        # 1. Forward Lookup
        if definition:
            forward_messages = [
                Templates.forward_lookup[0], # System message
                {"role": "User", "content": Templates.forward_lookup[1]["content"].replace("{{word}}", word)},
                {"role": "Chatbot", "content": Templates.forward_lookup[2]["content"].replace("{{definition}}", definition)}
            ]
            f.write(json.dumps({"messages": forward_messages}, ensure_ascii=False) + '\n')

        # 2. Reverse Lookup
        if definition:
            reverse_messages = [
                Templates.reverse_lookup[0], # System message
                {"role": "User", "content": Templates.reverse_lookup[1]["content"].replace("{{definition}}", definition)},
                {"role": "Chatbot", "content": Templates.reverse_lookup[2]["content"].replace("{{word}}", word)}
            ]
            f.write(json.dumps({"messages": reverse_messages}, ensure_ascii=False) + '\n')

        # 3. Phonetic
        if phonetics:
            phonetic_messages = [
                Templates.phonetic[0], # System message
                {"role": "User", "content": Templates.phonetic[1]["content"].replace("{{word}}", word)},
                {"role": "Chatbot", "content": Templates.phonetic[2]["content"].replace("{{phonetics}}", phonetics)}
            ]
            f.write(json.dumps({"messages": phonetic_messages}, ensure_ascii=False) + '\n')

        # 4. Etymology
        if origin and str(origin) != "nan": # Checking against string "nan" might be redundant now with fillna('')
            etymology_messages = [
                Templates.etymology[0], # System message
                {"role": "User", "content": Templates.etymology[1]["content"].replace("{{word}}", word)},
                {"role": "Chatbot", "content": Templates.etymology[2]["content"].replace("{{word}}", word).replace("{{origin}}", origin)}
            ]
            f.write(json.dumps({"messages": etymology_messages}, ensure_ascii=False) + '\n')

        # 5. Alphabetical Indexing
        if alphabet: # Check if alphabet index exists
             alphabetical_messages = [
                Templates.alphabetical_indexing[0], # System message
                {"role": "User", "content": Templates.alphabetical_indexing[1]["content"].replace("{{word}}", word)},
                {"role": "Chatbot", "content": Templates.alphabetical_indexing[2]["content"].replace("{{alphabet}}", alphabet)}
            ]
             f.write(json.dumps({"messages": alphabetical_messages}, ensure_ascii=False) + '\n')

print(f"Finished generating {output_file}")
