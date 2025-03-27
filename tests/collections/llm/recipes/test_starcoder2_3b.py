# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import nemo_run as run
import pytest

from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.peft import PEFT_STR2CLS
from nemo.collections.llm.recipes import starcoder2_3b
from nemo.lightning import Trainer


class TestStarcoder2_3B:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return starcoder2_3b

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert isinstance(model_config, run.Config)

    def test_pretrain_recipe(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain
        assert isinstance(recipe.model, run.Config)
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == MockDataModule
        assert recipe.data.seq_length == 4096
        assert recipe.data.global_batch_size == 32
        assert recipe.data.micro_batch_size == 2

        # Check parallelism settings
        assert recipe.trainer.strategy.tensor_model_parallel_size == 2
        assert recipe.trainer.strategy.pipeline_model_parallel_size == 1
        assert recipe.trainer.strategy.pipeline_dtype is None

        # Check trainer settings
        assert recipe.trainer.max_steps == 300000
        assert recipe.trainer.accumulate_grad_batches == 1
        assert recipe.trainer.val_check_interval == 1000
        assert recipe.trainer.limit_test_batches == 32
        assert recipe.trainer.limit_val_batches == 32
        assert recipe.trainer.log_every_n_steps == 10

    def test_finetune_recipe(self, recipe_module):
        recipe = recipe_module.finetune_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune
        assert isinstance(recipe.model, run.Config)
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == SquadDataModule

        # Check PEFT configuration
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == PEFT_STR2CLS['lora']
        assert recipe.optim.config.lr == 1e-4

    def test_finetune_recipe_with_dora(self, recipe_module):
        recipe = recipe_module.finetune_recipe(peft_scheme='dora')
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == PEFT_STR2CLS['dora']
        assert recipe.optim.config.lr == 1e-4

    def test_finetune_recipe_without_peft(self, recipe_module):
        recipe = recipe_module.finetune_recipe(peft_scheme=None)
        assert not hasattr(recipe, 'peft') or recipe.peft is None
        assert recipe.trainer.strategy.tensor_model_parallel_size == 2
        assert recipe.optim.config.lr == 5e-6

    def test_finetune_recipe_with_invalid_peft(self, recipe_module):
        with pytest.raises(ValueError, match="Unrecognized peft scheme: invalid_scheme"):
            recipe_module.finetune_recipe(peft_scheme="invalid_scheme")

    def test_finetune_recipe_with_packed_sequence(self, recipe_module):
        recipe = recipe_module.finetune_recipe(packed_sequence=True)
        assert hasattr(recipe.data, 'packed_sequence_specs')
