import random
import re
import json
from typing import Optional
from datasets import load_dataset

from ayamytk.test.bench.common import (
    HTML_JINJA,
    normalize_extracted_answer,
    normalize_response,
    map_with_progress,
    aggregate_results,
    jinja_env,
)
from ayamytk.test.bench.models import (
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
    EvalResult,
)

QUERY_TEMPLATE_MULTICHOICE_3 = """
အောက်ပါ မေးခွန်းအတွက် အဖြေမှန်ကို ရွေးပါ။ သင့်အဖြေ၏ နောက်ဆုံးစာကြောင်းသည် 'အဖြေ: $အစဉ်' (ဥပမာ- 'အဖြေ: က') ဖြစ်ရမည်။ $အစဉ် သည် (က၊ ခ၊ ဂ) တစ်ခုဖြစ်ရမည်။ အဖြေမပေးမီ အဆင့်ဆင့်စဉ်းစားပြီး ရွေးချယ်ပါ။

{question}

(က) {option_a}
(ခ) {option_b}
(ဂ) {option_c}
""".strip()

MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = (
    "(?i){head}[ \t]*(?:\\()?([က-ဃ]|[က]|[ခ]|[ဂ]|[ဃ]|{answer})(?:\\))?"
)

TRUE_FALSE_TEMPLATE = """
အောက်ပါကို မှားလျှင် (မှား)၊ မှန်လျှင် (မှန်) ဟု ဖြေဆိုပါ။ သင့်အဖြေ၏ နောက်ဆုံးစာကြောင်းသည် ဤပုံစံဖြစ်သင့်သည် - 'အဖြေ: မှန်' သိုမဟုတ် 'အဖြေ: မှား' (ကိုးကားချက်အမှတ်အသား မပါရ)။ အဖြေမပေးခင် အဆင့်ဆင့်စဉ်းစားပါ။

{question}
""".strip()

TRUE_FALSE_ANSWER_PATTERN_TEMPLATE = "(?i){}[ \t]*(?:\\()?(မှန်|မှား)(?:\\))?"

FILL_IN_BLANK_TEMPLATE = """
အောက်ပါစာကြောင်းရှိ ကွက်လပ်ကိုဖြည့်ပါ။ သင့်အဖြေ၏ နောက်ဆုံးစာကြောင်းသည် ဤပုံစံဖြစ်သင့်သည် - 'အဖြေ: $စာလုံး' (ကိုးကားချက်အမှတ်အသား မပါရ) စာလုံး သည် ကွက်လပ်အတွက် အသင့်တော်ဆုံး စကားလုံးဖြစ်သင့်သည်။ အဖြေမပေးခင် အဆင့်ဆင့်စဉ်းစားပါ။

{question}
""".strip()

SHORT_QNA_TEMPLATE = """
အောက်ပါမေးခွန်းကို အဆင့်ဆင့်စဉ်းစား၍ ဖြေဆိုပါ။

{question}
""".strip()


FILL_IN_BLANK_ANSWER_PATTERN_TEMPLATE = "(?i){head}[ \t]*(?:\\()?(.*?)(?:\\))?$"


def format_question(row):
    if row["type"] == "MCQ":
        return QUERY_TEMPLATE_MULTICHOICE_3.format(**row)
    elif row["type"] == "TOF":
        return TRUE_FALSE_TEMPLATE.format(**row)
    elif row["type"] == "FIB":
        return FILL_IN_BLANK_TEMPLATE.format(**row)
    elif (
        row["type"] == "SHORT_QNA"
        or row["type"] == "LONG_QNA"
        or row["type"] == "METAPHOR_QNA"
    ):
        return SHORT_QNA_TEMPLATE.format(**row)
    else:
        raise ValueError(f"Unknown question type: {row['type']}")


# Prompt template for evaluating Burmese questions and responses
GRADER_TEMPLATE = """
You are an expert Burmese‐language examiner. Evaluate the following:

Question:
{question}

Expected Answer:
{target}

Model Response:
{predicted_answer}

Using the scales below, assign each criterion a score and give a one‐sentence justification. Then output just the scores in a simple format.

1. Content Relevancy (0-3)
   0: Completely irrelevant
   1: Tangentially relevant
   2: Mostly relevant
   3: Fully relevant

2. Register Appropriateness (0-3)
   0: Completely inappropriate register
   1: Some register issues
   2: Mostly appropriate
   3: Fully appropriate (honorifics, particles, literary style)

3. Grammatical and Syntactic Competence (1-5)
   1: Very poor
   2: Poor
   3: Fair
   4: Good
   5: Excellent

Output format:

Content Relevancy: <0-3>
Register Appropriateness: <0-3>
Grammatical and Syntactic Competence: <1-5>
"""


ANSWER_REGEX = "အဖြေ\\s*:(?:\\n{0,2})?"


class ExamEval(Eval):
    def __init__(
        self,
        grader_model,
        num_examples: Optional[int] = None,
        language: str = "EN-US",
        filter_type=None,
    ):
        if language != "MYA":
            raise ValueError(f"Language {language} not supported")

        # Load dataset and convert to list safely
        dataset = load_dataset("Rickaym/Myanmar-G12L-Benchmark", split="test")
        examples = list(dataset)
        if filter_type:
            examples = [r for r in examples if r["type"] == filter_type]
        # Sample if needed
        if num_examples and len(examples) > num_examples:
            examples = random.Random(0).sample(examples, num_examples)

        self.examples = examples
        self.grader_model = grader_model

    def grade_sample(self, question: str, target: str, predicted_answer: str) -> dict:
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            target=target,
            predicted_answer=predicted_answer,
        )
        prompt_messages = [
            self.grader_model._pack_message(content=grader_prompt, role="user")
        ]
        grade_output = self.grader_model(prompt_messages).response_text

        # Initialize empty dictionary for scores
        data = {}
        # Use regex to extract scores for each criterion
        content_match = re.search(r"Content Relevancy:\s*(\d+)", grade_output)
        register_match = re.search(r"Register Appropriateness:\s*(\d+)", grade_output)
        grammar_match = re.search(
            r"Grammatical and Syntactic Competence:\s*(\d+)", grade_output
        )

        # Extract values if matches found
        if content_match:
            data["Content Relevancy"] = int(content_match.group(1))
        if register_match:
            data["Register Appropriateness"] = int(register_match.group(1))
        if grammar_match:
            data["Grammatical and Syntactic Competence"] = int(grammar_match.group(1))

        # Expected keys and their valid ranges
        expected_schema = {
            "Content Relevancy": (0, 3),
            "Register Appropriateness": (0, 3),
            "Grammatical and Syntactic Competence": (1, 5),
        }

        # Validate presence and ranges
        for key, (min_val, max_val) in expected_schema.items():
            if key not in data:
                return {}
            value = data[key]
            if not isinstance(value, int) or not (min_val <= value <= max_val):
                return {}

        return data

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(content=format_question(row), role="user")
            ]
            response_text = normalize_response(sampler(prompt_messages).response_text)
            extracted_answer = None

            if row["type"] == "SHORT_QNA":
                data = self.grade_sample(row["question"], row["answer"], response_text)

                # Calculate total score as average of normalized scores
                content_score = data["Content Relevancy"] / 3.0
                register_score = data["Register Appropriateness"] / 3.0
                grammar_score = data["Grammatical and Syntactic Competence"] / 5.0
                score = (content_score + register_score + grammar_score) / 3.0

                # Create metrics dictionary with individual scores
                metrics = {
                    "SHORT_QNA": score,
                    "Content Relevancy": content_score,
                    "Register Appropriateness": register_score,
                    "Grammatical and Syntactic Competence": grammar_score,
                }

                # No extracted answer for SHORT_QNA
                extracted_answer = json.dumps(metrics)
            else:
                if row["type"] == "MCQ":
                    regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(
                        head=ANSWER_REGEX, answer=row["answer"]
                    )
                elif row["type"] == "TOF":
                    regex = TRUE_FALSE_ANSWER_PATTERN_TEMPLATE.format(ANSWER_REGEX)
                elif row["type"] == "FIB":
                    regex = FILL_IN_BLANK_ANSWER_PATTERN_TEMPLATE.format(
                        head=ANSWER_REGEX
                    )
                else:
                    regex = ANSWER_REGEX

                match = re.search(regex, response_text)
                if match:
                    try:
                        extracted_answer = match.group(1)
                    except IndexError:
                        pass

                if row["type"] == "MCQ" and extracted_answer:
                    extracted_answer = normalize_extracted_answer(extracted_answer)

                score = 1.0 if extracted_answer == row["answer"] else 0.0
                metrics = {row["type"]: score}

            html = jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html, score=score, metrics=metrics, convo=convo
            )

        results = map_with_progress(fn, self.examples)
        return aggregate_results(results)
