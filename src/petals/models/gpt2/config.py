import os
from typing import Optional, Union

from hivemind import get_logger
from transformers.models.gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from petals.client.config import ClientConfig
from petals.client.lm_head import LMHeadConfig
from petals.client.ptune import PTuneConfig
from petals.models.gpt2.block import WrappedGPT2Block

logger = get_logger(__name__)


class DistributedGPT2Config(GPT2Config, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedGPT2Block
    attn_class = GPT2Attention
    block_prefix = "h"

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path) + "-petals"
            dht_prefix = dht_prefix.replace(".", "-")
            logger.info(f"Using DHT prefix: {dht_prefix}")
        return super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)