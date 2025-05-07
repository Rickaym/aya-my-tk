import pandas as pd
import re
import random
from itertools import chain
from tqdm import tqdm


class Generator:
    def __init__(self, templates: list[list[list[dict]]], dataframe: pd.DataFrame):
        self.templates = templates
        self.dataframe = dataframe
        self.association = []
        self.logs = []

        for t in templates:
            combined_variables = chain.from_iterable(self.get_variables(e) for e in t)
            self.association.append((t, list(combined_variables)))

    def get_variables(self, template: list[dict]):
        variables = set()
        for msg in template:
            if "content" in msg and isinstance(msg["content"], str):
                # Find all variable patterns like {{var_name}} in the content
                var_matches = re.findall(r"{{(.*?)}}", msg["content"])
                for var_name in var_matches:
                    variables.add(var_name)
        return variables

    def generate(self):
        data = []
        for _, row in tqdm(self.dataframe.iterrows(), total=len(self.dataframe)):
            for templates, variables in self.association:
                template = random.choice(templates)
                is_valid = True
                for variable in variables:
                    if variable not in row:
                        self.logs.append(f"Variable {variable} not found in row {row}")
                        is_valid = False
                        break

                    value = row[variable]
                    if not value or pd.isna(value):
                        self.logs.append(
                            f"Value {value} is not valid for variable {variable}"
                        )
                        is_valid = False
                        break

                    template = [
                        {
                            **t,
                            "content": t["content"].replace(
                                "{{" + variable + "}}", str(value)
                            ),
                        }
                        for t in template
                    ]

                if is_valid:
                    data.append(
                        {
                            "messages": template,
                        }
                    )
        print("\n".join(self.logs))
        return data
