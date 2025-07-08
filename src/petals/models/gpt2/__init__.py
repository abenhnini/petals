from petals.models.gpt2.block import WrappedGPT2Block
from petals.models.gpt2.config import DistributedGPT2Config
from petals.models.gpt2.model import (
    DistributedGPT2ForCausalLM,
    DistributedGPT2Model,
)
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedGPT2Config,
    model=DistributedGPT2Model,
    model_for_causal_lm=DistributedGPT2ForCausalLM,
)