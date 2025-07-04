import pandas as pd

# အောက်တွင် လုပ်ဆောင်ရန် လုပ်ငန်းတစ်ခုကို ဖော်ပြထားပြီး၊ နောက်ထပ် အခြေအနေကို ပေးထားသော အချက်အလက်နှင့် တွဲဖက်ထားပါသည်။ တောင်းဆိုချက်ကို သင့်လျော်စွာ ဖြည့်စွက်ပေးသော တုံ့ပြန်မှုတစ်ခု ရေးပါ။

# ### ညွှန်ကြားချက်:
# {instruction}

# ### အချက်အလက်:
# {input}

# ### တုံ့ပြန်မှု:


def alpaca_formatter(row: pd.Series) -> str:
    """
    Default formatter for Alpaca-style datasets

    Args:
        row: Pandas Series representing a row of data

    Returns:
        Formatted string for the user message
    """
    instruction = str(row.get("instruction", ""))
    input_text = str(row.get("input", ""))

    if input_text and input_text.strip():
        return f"""အောက်တွင် လုပ်ဆောင်ရန် လုပ်ငန်းတစ်ခုကို ဖော်ပြထားပြီး၊ နောက်ထပ် အခြေအနေကို ပေးထားသော အချက်အလက်နှင့် တွဲဖက်ထားပါသည်။ တောင်းဆိုချက်ကို သင့်လျော်စွာ ဖြည့်စွက်ပေးသော တုံ့ပြန်မှုတစ်ခု ရေးပါ။

### ညွှန်ကြားချက်:
{instruction}

### အချက်အလက်:
{input_text}

### တုံ့ပြန်မှု:"""
    else:
        return f"""အောက်တွင် လုပ်ဆောင်ရန် လုပ်ငန်းတစ်ခုကို ဖော်ပြထားပါသည်။ တောင်းဆိုချက်ကို သင့်လျော်စွာ ဖြည့်စွက်ပေးသော တုံ့ပြန်မှုတစ်ခု ရေးပါ။

### ညွှန်ကြားချက်:
{instruction}

### တုံ့ပြန်မှု:"""


def simple_formatter(row: pd.Series) -> str:
    """
    Simple formatter that just uses the 'input' or 'prompt' column

    Args:
        row: Pandas Series representing a row of data

    Returns:
        The input text as-is
    """
    return row.get("input", row.get("prompt", ""))
