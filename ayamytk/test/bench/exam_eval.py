import random
import re
from typing import Optional
import pandas

from common import (
    HTML_JINJA,
    MULTILINGUAL_ANSWER_REGEXES,
    normalize_extracted_answer,
    normalize_response,
    map_with_progress,
    aggregate_results,
    QUERY_TEMPLATE_MULTICHOICE,
    jinja_env,
)
from models import Eval, EvalResult, SamplerBase, SingleEvalResult

QUERY_TEMPLATE_MULTICHOICE_3 = "\n".join(QUERY_TEMPLATE_MULTICHOICE.split("\n")[:-1])

MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = (
    "(?i){head}[ \t]*(?:\\()?([က-ဃ]|[က]|[ခ]|[ဂ]|[ဃ]|{answer})(?:\\))?"
)

TRUE_FALSE_TEMPLATE = """
အောက်ပါကို မှားလျှင် (မှား)၊ မှန်လျှင် (မှန်) ဟု ဖြေဆိုပါ။ သင့်အဖြေ၏ နောက်ဆုံးစာကြောင်းသည် ဤပုံစံဖြစ်သင့်သည် - 'အဖြေ: မှန်' သိုမဟုတ် 'အဖြေ: မှား' (ကိုးကားချက်အမှတ်အသား မပါဘဲ)။ အဖြေမပေးခင် အဆင့်ဆင့်စဉ်းစားပါ။

{question}
""".strip()

TRUE_FALSE_ANSWER_PATTERN_TEMPLATE = "(?i){}[ \t]*(?:\\()?(မှန်|မှား)(?:\\))?"

FILL_IN_BLANK_TEMPLATE = """
အောက်ပါစာကြောင်းရှိ ကွက်လပ်ကိုဖြည့်ပါ။ သင့်အဖြေ၏ နောက်ဆုံးစာကြောင်းသည် ဤပုံစံဖြစ်သင့်သည် - 'အဖြေ: $စာလုံး' (ကိုးကားချက်အမှတ်အသား မပါဘဲ) စာလုံး သည် ကွက်လပ်အတွက် အသင့်တော်ဆုံး စကားလုံးဖြစ်သင့်သည်။ အဖြေမပေးခင် အဆင့်ဆင့်စဉ်းစားပါ။

{question}
""".strip()

FILL_IN_BLANK_ANSWER_PATTERN_TEMPLATE = "(?i){head}[ \t]*(?:\\()?(.*?)(?:\\))?$"


def format_question(row):
    if row["Type"] == "Multiple Choice":
        return QUERY_TEMPLATE_MULTICHOICE_3.format(**row)
    elif row["Type"] == "True/False":
        return TRUE_FALSE_TEMPLATE.format(**row)
    elif row["Type"] == "Fill in the Blank":
        return FILL_IN_BLANK_TEMPLATE.format(**row)
    else:
        raise ValueError(f"Unknown question type: {row['Type']}")


class ExamEval(Eval):
    def __init__(self, num_examples: Optional[int] = None, language: str = "EN-US"):
        if language != "MYA":
            raise ValueError(f"Language {language} not supported")

        TRUE_OR_FALSE = pandas.read_csv("G12-Exam/tof_qna.csv")
        TRUE_OR_FALSE["Type"] = "True/False"

        MULTIPLE_CHOICE = pandas.read_csv("G12-Exam/multiple_choice_qna.csv")
        MULTIPLE_CHOICE["Type"] = "Multiple Choice"

        FILL_IN_BLANK = pandas.read_csv("G12-Exam/fib.csv")
        FILL_IN_BLANK["Type"] = "Fill in the Blank"

        # Combine all datasets
        combined_df = pandas.concat([TRUE_OR_FALSE, MULTIPLE_CHOICE, FILL_IN_BLANK])
        examples = [row.to_dict() for _, row in combined_df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(content=format_question(row), role="user")
            ]
            response_text = normalize_response(sampler(prompt_messages))
            extracted_answer = None

            for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
                if row["Type"] == "Multiple Choice":
                    regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(
                        head=answer_regex, answer=row["Answer"]
                    )
                elif row["Type"] == "True/False":
                    regex = TRUE_FALSE_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
                elif row["Type"] == "Fill in the Blank":
                    regex = FILL_IN_BLANK_ANSWER_PATTERN_TEMPLATE.format(
                        head=answer_regex
                    )
                else:
                    regex = answer_regex

                match = re.search(regex, response_text)
                if match:
                    extracted_answer = match.group(1)
                    break

            if row["Type"] == "Multiple Choice" and extracted_answer:
                extracted_answer = normalize_extracted_answer(extracted_answer)

            score = 1.0 if extracted_answer == row["Answer"] else 0.0
            html = jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html, score=score, metrics={row["Type"]: score}, convo=convo
            )

        results = map_with_progress(fn, self.examples)
        return aggregate_results(results)
