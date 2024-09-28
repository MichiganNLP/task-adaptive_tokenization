
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
"""reddit_mhp dataset."""


import json
import os
import datasets



_DESCRIPTION = """ FutureWarning
"""
_CITATION = """ null """
_URLs = {
    "train": "https://huggingface.co/datasets/siyangliu/reddit_mhp/resolve/main/train.json",
    "valid": "https://huggingface.co/datasets/siyangliu/reddit_mhp/resolve/main/valid.json",
    "test": "https://huggingface.co/datasets/siyangliu/reddit_mhp/resolve/main/test.json",
}



class redditMHP(datasets.GeneratorBasedBuilder):
    """redditMHP dataset."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="plain_text",
            description="plain text",
            version=VERSION,
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "questionID": datasets.Value("string"),
                    "description": datasets.Value("string"),
                    "topic": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "answerID": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://huggingface.co/datasets/siyangliu/reddit_mhp",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URLs)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"]
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"]
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["valid"]
                },
            ),
        ]

    def _generate_examples(self, filepath, label_filepath=None, strategy=False):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as input_file:
            dataset = json.load(input_file)
            idx = 0 
            for meta_data in dataset:
                yield idx, {"question": meta_data["question"], "description": meta_data["description"], "questionID":meta_data['post_id'], "answerID": meta_data["comment_id"], "answer": meta_data["answer"], "topic":meta_data["topic"]}
                idx += 1
        
            