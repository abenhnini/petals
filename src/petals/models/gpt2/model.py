from typing import Optional

import hivemind
import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2 import GPT2ForCausalLM, GPT2Model, GPT2PreTrainedModel

from petals.client.from_pretrained import FromPretrainedMixin
from petals.client.lm_head import LMHead
from petals.client.ptune import PTuneMixin
from petals.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from petals.client.remote_sequential import RemoteSequential
from petals.models.gpt2.config import DistributedGPT2Config

logger = get_logger(__name__)


class DistributedGPT2Model(FromPretrainedMixin, PTuneMixin, GPT2Model):
    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^h\."]

    config_class = DistributedGPT2Config

    def __init__(self, config: DistributedGPT2Config, *, dht: Optional[hivemind.DHT] = None):
        n_layer, config.n_layer = config.n_layer, 0  # Prevent initialization
        super().__init__(config)
        assert len(self.h) == 0
        config.n_layer = n_layer

        self.h = RemoteSequential(config, dht=dht)

        self.requires_grad_(False)  # Forbid accumulate grads for embeddings
        self.init_prompts(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RemotePastKeyValues] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_key_values = RemotePastKeyValues()

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        output_shape = input_shape + (hidden_states.size(-1),)

        hidden_states = self.h(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )


class DistributedGPT2ForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, GPT2ForCausalLM):
    _keys_to_ignore_on_load_missing = DistributedGPT2Model._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_missing += [r"^lm_head\."]
    _keys_to_ignore_on_load_unexpected = DistributedGPT2Model._keys_to_ignore_on_load_unexpected

    config_class = DistributedGPT2Config

    def __init__(self, config: DistributedGPT2Config):
        GPT2PreTrainedModel.__init__(self, config)
        self.transformer = DistributedGPT2Model(config)
        self.lm_head = LMHead(config)

        self.post_init()