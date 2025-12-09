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

import inspect

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score import default_compute_score
import asyncio
import json
import logging
import os
import torch
import aiohttp
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

@register("debate_reward_loop_manager")
class DebateRewardLoopManager(RewardLoopManagerBase):
    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        player_signs = data_item.non_tensor_batch["extra_fields"]["player_signs"][:valid_response_length]
        tsigns = torch.as_tensor([0]+player_signs+[0])
        diffs = tsigns[1:] - tsigns[:-1]
        # diffs = 1 implies player A start or B end, diffs = -1 => player B start or end
        change_pos = torch.nonzero(diffs, as_tuple=False).flatten()
        start_pos = change_pos[::2]
        end_pos = change_pos[1::2]

        debate_topic = data_item.non_tensor_batch["topic"]
        # "are apples or oranges the superior fruit?"
        position_1 = data_item.non_tensor_batch["position_1"]
        position_2 = data_item.non_tensor_batch["position_2"]

        argument_ids = []
        for start, end in zip(start_pos, end_pos):
            argument_ids.append(valid_response_ids[start:end])

        argument_strs = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.batch_decode(argument_ids, skip_special_tokens=True)
        )
        result = await self.compute_score(
            data_source=data_source,
            argument_strs=argument_strs,
            topic=debate_topic,
            position_1=position_1,
            position_2=position_2,
            reward_router_address=self.reward_router_address,
            reward_model_tokenizer=self.reward_model_tokenizer,
        )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
