# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the GSM8k dataset to parquet format
"""

import os

import datasets

data = [
    {
        "topic": "are apples or oranges the superior fruit?",
        "position_1": "apples are the superior fruit",
        "position_2": "oranges are the superior fruit",
    },
    {
        "topic": "are cats or dogs the better pet?",
        "position_1": "cats are the better pet",
        "position_2": "dogs are the better pet",
    },
    {
        "topic": "which political party has better economic policies in the United States?",
        "position_1": "the Democratic Party has better economic policies",
        "position_2": "the Republican Party has better economic policies",
    },
    {
        "topic": "is China or the United States more likely to gain the economic upperhand in the next 10 years?",
        "position_1": "China is likely to gain the economic upperhand in the next 10 years",
        "position_2": "the United States is likely to gain the economic upperhand in the next 10 years",
    },
    {
        "topic": "is the death penalty effective in deterring crime?",
        "position_1": "the death penalty is effective in deterring crime",
        "position_2": "the death penalty is not effective in deterring crime",
    },
    {
        "topic": "does increasing the number of countries with nuclear weapons make the world safer?",
        "position_1": "increasing the number of countries with nuclear weapons makes the world safer",
        "position_2": "increasing the number of countries with nuclear weapons makes the world more dangerous",
    },
    {
        "topic": "are school vouchers good for education?",
        "position_1": "school vouchers are good for education",
        "position_2": "school vouchers are bad for education",
    },
    {
        "topic": "is social media good for society?",
        "position_1": "social media is good for society",
        "position_2": "social media is bad for society",
    }
]

# flip positions for more data
extra_data = [
    {"topic": d["topic"], "position_1": d["position_2"], "position_2": d["position_1"]} for d in data
]

data.extend(extra_data)

def add_field(data, field, value):
    for d in data:
        d[field] = value

add_field(data, "agent_name", "debate_agent")
add_field(data, "data_source", "debate_topics")

dataset = datasets.Dataset.from_list(data)
dataset.to_parquet("./train.parquet")