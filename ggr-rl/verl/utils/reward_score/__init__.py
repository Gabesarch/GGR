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
# from . import gsm8k, math, prime_math, prime_code

def default_compute_score(*args, **kwargs):
    raise ValueError("Set reward.custom_reward_function to the GGR reward module")


def get_default_compute_score(reward_name=None):
    return default_compute_score


_default_compute_score = default_compute_score
__all__ = ["default_compute_score", "get_default_compute_score"]
