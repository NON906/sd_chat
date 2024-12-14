from typing import List
from pydantic import BaseModel

class CheckPointSettings(BaseModel):
    name: str
    prompt: str = ''
    negative_prompt: str = ''
    cfg_scale: float = 7.0
    sampler_name: str = 'Euler a'
    steps: int = 20
    width: int = 512
    height: int = 512
    enable_hr: bool = False
    hr_upscaler: str = "R-ESRGAN 4x+"
    hr_second_pass_steps: int = 20
    hr_resize_x: str = 1024
    hr_resize_y: str = 1024
    denoising_strength: float = 0.4

class LoraSettings(BaseModel):
    name: str
    trigger_words: List[str] = []
    weight: float = 1.0
    caption: str = ""
