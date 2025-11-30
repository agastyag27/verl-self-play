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
import copy
import logging
import os
from re import S
from typing import Any
from uuid import uuid4
import asyncio

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("debate_agent")
class DebateAgent(AgentLoopBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.num_turns = self.config.actor_rollout_ref.rollout.num_turns
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        debate_topic = kwargs["topic"]
        # "are apples or oranges the superior fruit?"
        position_1 = kwargs["position_1"]
        position_2 = kwargs["position_2"]
        # "apples are the superior fruit"
        # dataset should contain both orders

        metrics = {}
        request_id = uuid4().hex

        messages = [
            {"role": "system",
            "content": f"You are a pro debater who must argue for both sides of the following debate: {debate_topic}. Each argument should respond to previous points and further your asigned side's case. Keep all arguments concise (max 4 sentences), and make sure you are arguing only for your current assigned side."}
        ]

        tasks = []

        tasks.append(self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        ))

        tasks.append(self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"Argue for the following position: {position_1}"}], add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        ))

        tasks.append(self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"Argue for the following position: {position_2}"}], add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        ))

        results = await asyncio.gather(*tasks)

        system_ids = results[0]
        user_ids = results[1:]

        cur_ids = [id for id in system_ids]

        response_mask = []
        response_logprobs = []
        player_signs = []

        budget_per_turn = (self.response_length - self.num_turns * (len(user_ids[1]) + len(user_ids[0]))) // (2 * self.num_turns)
        assert budget_per_turn > 0, f"Budget per turn is {budget_per_turn}, which is not positive"

        for turn in range(2*self.num_turns):
            cur_ids.extend(user_ids[turn%2])
            player_signs.extend([0]*len(user_ids[turn%2]))
            response_mask.extend([0]*len(user_ids[turn%2]))
            response_logprobs.extend([0]*len(user_ids[turn%2]))
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=cur_ids, sampling_params=sampling_params
                )
            
            response_ids = output.token_ids[: budget_per_turn]
            cur_ids.extend(response_ids)
            cur_sign = -1 if turn % 2 else 1
            player_signs.extend([cur_sign]*len(response_ids))
            response_mask.extend([1]*len(response_ids))
            if output.log_probs:
                response_logprobs.extend(output.log_probs[: budget_per_turn])

        response_ids = cur_ids[len(system_ids):]

        output = AgentLoopOutput(
            prompt_ids=system_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs if response_logprobs else None,
            multi_modal_data={},
            num_turns=1 + 4 * self.num_turns,
            metrics=metrics,
            extra_fields={"player_signs": player_signs}
        )
        return output
