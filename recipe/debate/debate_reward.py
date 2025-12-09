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
import math
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

JUDGE_SAMPLING_PARAMS = {
    "max_tokens": 5,       # Scan first 5 tokens for a valid answer
    "skip_special_tokens": True,
}

OTHER_PARAMS = {
    "top_logprobs_num": 20,        # Get top-20 candidates to find "1" and "2"
    "return_logprob": True, 
}

async def generate_aiohttp(router_address: str, prompt_ids: list[int], sampling_params: dict):
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
        **OTHER_PARAMS,
    }
    # print(f"Payload: {payload}")
    url = f"http://{router_address}/generate"
    try:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        async with session.post(url, json=payload) as resp:
            output = await resp.text()
            try:
                output = json.loads(output)
                return output
            except Exception:
                logger.error(f"Failed to parse JSON response: {output}")
                return {}
    finally:
        await session.close()

# Define variants (e.g. "1", " 1", "2", " 2")
TARGET_MAP = {"1": 1, " 1": 1, "2": 2, " 2": 2}

async def compute_score_debate(
    data_source: str,
    argument_strs: list[str],
    topic: str,
    position_1: str,
    position_2: str,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer,
) -> dict:
    """Compute the reward score for Debate."""
    loop = asyncio.get_running_loop()

    # --- 1. PRE-COMPUTE TOKEN IDs ---
    # We cannot rely on text matching because SGLang returns None for text.
    # We must match against the integer Token IDs.
    
    # We define the strings we want to look for
    target_strings = {
        "1": 1, 
        " 1": 1, 
        "2": 2, 
        " 2": 2
    }
    
    # Convert string keys to integer IDs using the tokenizer
    # map: {15: 1, 220: 2, ...} (Example IDs)
    target_token_map = {}
    
    for text, value in target_strings.items():
        # encode(add_special_tokens=False) ensures we get just the raw ID 
        # without <|im_start|> etc.
        try:
            ids = reward_model_tokenizer.encode(text, add_special_tokens=False)
            if ids:
                # We take the last ID if it produces multiple (e.g. " 1" -> [" ", "1"])
                # usually for digits it's a single token.
                target_id = ids[-1]
                target_token_map[target_id] = value
        except Exception as e:
            logger.warning(f"Failed to encode target token '{text}': {e}")

    # --------------------------------

    debate_strs = []
    for i, argument_str in enumerate(argument_strs):
        debate_strs.append(f"Person {i%2+1}'s argument: {argument_str}\n\n")

    debate_strs.append(f"Decide which side you believe more now that you have heard both arguments. Output 1 if you believe {position_1} and 2 if you believe {position_2}. Output ONLY this single character.")  
    messages = [
        {"role": "system", "content": f"You observed the following debate on the topic: {topic}."},
        {"role": "user", "content": ''.join(debate_strs)}
    ]

    judge_ids = await loop.run_in_executor(
        None,
        lambda: reward_model_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        ),
    )
    
    # ... (Network call remains the same) ...
    judge_outputs = await generate_aiohttp(
        router_address=reward_router_address,
        prompt_ids=judge_ids,
        sampling_params=JUDGE_SAMPLING_PARAMS,
    )

    # Use 'output_top_logprobs' as seen in your logs
    meta_info = judge_outputs.get("meta_info", {})
    all_steps_logprobs = meta_info.get("output_top_logprobs", [])

    if not all_steps_logprobs:
        # print(f"Judge outputs: {judge_outputs}") # Optional debug
        logger.error("No logprobs returned from SGLang")
        return {"score": 0.0, "judge_succesful": False}

    # --- SCORE CALCULATION ---
    prob_1 = 0.0
    prob_2 = 0.0
    found_any = False

    # Scan the first few generated steps
    for step_logprobs in all_steps_logprobs[:10]: 
        step_p1 = 0.0
        step_p2 = 0.0
        step_has_target = False
        
        # logprob structure: [logprob_float, token_id_int, token_text_str_or_None]
        for logprob, token_id, _ in step_logprobs:
            
            # --- FIX: Match by ID, not Text ---
            val = target_token_map.get(token_id, None)
            
            if val == 1:
                step_p1 += math.exp(logprob)
                step_has_target = True
            elif val == 2:
                step_p2 += math.exp(logprob)
                step_has_target = True
        
        # If we found relevant tokens in this step, use them and STOP scanning.
        if step_has_target:
            prob_1 = step_p1
            prob_2 = step_p2
            found_any = True
            break 
    
    if not found_any:
        return {"score": 0.0, "judge_succesful": False}

    # SAFETY: Add epsilon to prevent log(0) or division by zero
    epsilon = 1e-9
    safe_p1 = prob_1 + epsilon
    safe_p2 = prob_2 + epsilon

    # FORMULA: log(p1 / p2)
    raw_score = math.log(safe_p1 / safe_p2)

    # SAFETY: Clip the reward
    CLIP_VAL = 10.0
    reward_score = max(min(raw_score, CLIP_VAL), -CLIP_VAL)

    return {"score": reward_score, "judge_succesful": True}