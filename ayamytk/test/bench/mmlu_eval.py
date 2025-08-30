"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import re
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
from ayamytk.test.bench.models import Eval, EvalResult, SamplerBase, SingleEvalResult

subject2category = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}

QUERY_TEMPLATE_MULTICHOICE = """
အောက်ပါ မေးခွန်းအတွက် အဖြေမှန်ကို ရွေးပါ။ သင့်အဖြေ၏ နောက်ဆုံးစာကြောင်းသည် 'အဖြေ: $အစဉ်' (ဥပမာ- 'အဖြေ: က') ဖြစ်ရမည်။ $အစဉ် သည် (က၊ ခ၊ ဂ၊ ဃ) တစ်ခုဖြစ်ရမည်။ အဖြေမပေးမီ အဆင့်ဆင့်စဉ်းစားပြီး ရွေးချယ်ပါ။

{question}

(က) {option_a}
(ခ) {option_b}
(ဂ) {option_c}
(ဃ) {option_d}
""".strip()

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = (
    "(?i){}[ \t]*(?:\\()?([က-ဃ]|[က]|[ခ]|[ဂ]|[ဃ])(?:\\))?"
)
# All the different ways "Answer" is written in different languages
MULTILINGUAL_ANSWER_REGEXES = [
    "အဖြေ\\s*:(?:\\n{0,2})?",
]


def format_multichoice_question(row):
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


class MMLUEval(Eval):
    def __init__(self, num_examples: Optional[int] = None, language: str = "EN-US", num_threads: int = 50):
        if language != "MYA":
            raise ValueError("Language must be MYA")

        df = load_dataset("Rickaym/Burmese-MMLU-Lite", split="test")
        examples = list(df)
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.num_threads = num_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(row), role="user"
                )
            ]
            response_text = normalize_response(sampler(prompt_messages).response_text)
            extracted_answer = None
            for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
                regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
                match = re.search(regex, response_text)
                if match:
                    extracted_answer = normalize_extracted_answer(match.group(1))
                    break
            score = 1.0 if extracted_answer == row["answer"] else 0.0
            html = jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            category = subject2category.get(
                row.get("sample_id", "").split("/")[0], "other"
            )
            return SingleEvalResult(
                html=html, score=score, metrics={category: score}, convo=convo
            )

        results = map_with_progress(fn, self.examples, num_threads=self.num_threads)
        return aggregate_results(results)
