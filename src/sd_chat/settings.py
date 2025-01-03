from typing import List, Dict
from pydantic import BaseModel

class LoraSettings(BaseModel):
    name: str
    trigger_words: List[str] = []
    weight: float = 1.0
    caption: str = ''
    base_model: str = ''

class CheckPointSettings(BaseModel):
    name: str
    caption: str = ''
    prompt: str = ''
    negative_prompt: str = ''
    cfg_scale: float = 7.0
    sampler_name: str = 'Euler a'
    steps: int = 20
    width: int = 512
    height: int = 512
    clip_skip: int = 1
    base_model: str = ''
    not_installed: bool = False
    version_id: int = -1
    loras: Dict[str, LoraSettings] = {}