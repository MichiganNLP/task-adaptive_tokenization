
# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PsyQA dataset."""


import json
import os
import datasets



_DESCRIPTION = """ FutureWarning
"""
_CITATION = """ null """
_URLs = {
    "train": "https://huggingface.co/datasets/siyangliu/PsyQA/resolve/main/train.json",
    "valid": "https://huggingface.co/datasets/siyangliu/PsyQA/resolve/main/valid.json",
    "test": "https://huggingface.co/datasets/siyangliu/PsyQA/resolve/main/test.json",
    "train_translated": "https://huggingface.co/datasets/siyangliu/PsyQA/resolve/main/train_translated.json",
    "valid_translated": "https://huggingface.co/datasets/siyangliu/PsyQA/resolve/main/valid_translated.json",
    "test_translated": "https://huggingface.co/datasets/siyangliu/PsyQA/resolve/main/test_translated.json"
}

_STRATEGY={"Approval and Reassurance": "[AR]",
           "Interpretation": "[IN]",
           "Self-disclosure": "[SELF]",
           "Direct Guidance": "[DG]",
           "Others": "[OT]",
           "Restatement": "[RES]",
           "Information": "[INFO]"}


class PsyQA(datasets.GeneratorBasedBuilder):
    """PsyQA dataset."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="wo strategy",
            description="",
            version=VERSION,
        ),
        datasets.BuilderConfig(
            name="w strategy",
            description="",
            version=VERSION,
        ),
        datasets.BuilderConfig(
            name="translated",
            description="",
            version=VERSION,
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "questionID": datasets.Value("int16"),
                    "description": datasets.Value("string"),
                    "keywords": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "has_label": datasets.Value("bool"),
                    "reference":datasets.features.Sequence(datasets.Value("string"))
                    # "labels_sequence":datasets.features.Sequence(
                    #     {
                    #         "start": datasets.Value("int16"),
                    #         "end": datasets.Value("int16"),
                    #         "type": datasets.Value("string"),
                    #     }
                    # ),
                }
            ),
            supervised_keys=None,
            homepage="https://huggingface.co/datasets/siyangliu/PsyQA",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URLs)
        if self.config.name != "translated":
            
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": data_dir["train"],
                        "strategy": self.config.name == "w strategy"
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": data_dir["test"],
                        "strategy": self.config.name == "w strategy"
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": data_dir["valid"],
                        "strategy": self.config.name == "w strategy"
                    },
                ),
            ]
        else: 
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": data_dir["train_translated"]
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": data_dir["test_translated"]
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": data_dir["valid_translated"]
                    },
                ),
            ]
            

    def _generate_examples(self, filepath, label_filepath=None, strategy=False):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as input_file:
            dataset = json.load(input_file)
            idx = 0 
            for meta_data in dataset:
                reference = [ans["answer_text"] for ans in meta_data["answers"]]
                for ans in meta_data["answers"]:
                    if strategy and ans["labels_sequence"] is None:
                        continue
                    elif strategy and ans["labels_sequence"] is not None:
                        pieces = []
                        for label in ans["labels_sequence"]:
                            pieces.append(_STRATEGY[label["type"]]+ans["answer_text"][label["start"]:label["end"]])
                        ans_w_strategy = "".join(pieces)
                        yield idx, {"question": meta_data["question"], "description": meta_data["description"], "keywords": meta_data["keywords"], "answer": ans_w_strategy, \
                         "questionID": meta_data["questionID"], "has_label": ans["has_label"], "reference": reference}
                    else:        
                        yield idx, {"question": meta_data["question"], "description": meta_data["description"], "keywords": meta_data["keywords"], "answer": ans["answer_text"], \
                            "questionID": meta_data["questionID"], "has_label": ans["has_label"], "reference":reference}
                    idx += 1
        
            