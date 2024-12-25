import os
import sys
import datetime
import uuid
import asyncio
import threading
import glob
import gc
from contextlib import redirect_stdout
from PIL import PngImagePlugin
import aiohttp
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
)
import torch

from .settings import CheckPointSettings, LoraSettings

class SDAPI_Diffusers:
    image_threads = {}
    image_results = {}
    checkpoint_settings = None
    lora_settings = None
    pipeline = None

    def __init__(self, save_dir_path: str, checkpoints_dir_path: str | None = None, loras_dir_path: str | None = None):
        self.save_dir_path = save_dir_path
        if checkpoints_dir_path is None:
            self.checkpoints_dir_path = os.path.join(self.save_dir_path, 'models', 'StableDiffusion')
        else:
            self.checkpoints_dir_path = checkpoints_dir_path
        if loras_dir_path is None:
            self.loras_dir_path = os.path.join(self.save_dir_path, 'models', 'Lora')
        else:
            self.loras_dir_path = loras_dir_path

    async def get_checkpoints_dir_path(self):
        return self.checkpoints_dir_path
            
    async def get_loras_dir_path(self):
        return self.loras_dir_path

    def __txt2img(self, image_id: str, prompt: str, checkpoint_settings: CheckPointSettings, lora_settings: list):
        now_str = datetime.datetime.now().strftime('%Y-%m-%d')

        if self.pipeline is None or self.checkpoint_settings != checkpoint_settings or self.lora_settings != lora_settings:
            if self.pipeline is not None:
                del self.pipeline
                gc.collect()
                torch.cuda.empty_cache()

            self.checkpoint_settings = checkpoint_settings
            self.lora_settings = lora_settings

            target_list = glob.glob(os.path.join(self.checkpoints_dir_path, checkpoint_settings.name + '.*'))
            for target_name in target_list:
                if os.path.splitext(target_name)[1] == '.safetensors':
                    file_name = target_name

            if checkpoint_settings.base_model == 'SD 1.5':
                self.pipeline = StableDiffusionPipeline.from_single_file(
                    file_name,
                    torch_dtype=torch.float16,
                ).to("cuda")
            else:
                self.pipeline = StableDiffusionXLPipeline.from_single_file(
                    file_name,
                    torch_dtype=torch.float16,
                ).to("cuda")

            if checkpoint_settings.sampler_name == 'Euler a':
                self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)
            elif checkpoint_settings.sampler_name == 'DPM++ 2M':
                self.pipeline.scheduler = DPMSolverMultistepScheduler(self.pipeline.scheduler.config)
            elif checkpoint_settings.sampler_name == 'DPM++ SDE':
                self.pipeline.scheduler = DPMSolverSinglestepScheduler(self.pipeline.scheduler.config)
            elif checkpoint_settings.sampler_name == 'DPM2':
                self.pipeline.scheduler = KDPM2DiscreteScheduler(self.pipeline.scheduler.config)
            elif checkpoint_settings.sampler_name == 'DPM2 a':
                self.pipeline.scheduler = KDPM2AncestralDiscreteScheduler(self.pipeline.scheduler.config)
            elif checkpoint_settings.sampler_name == 'Euler':
                self.pipeline.scheduler = EulerDiscreteScheduler(self.pipeline.scheduler.config)
            elif checkpoint_settings.sampler_name == 'Heun':
                self.pipeline.scheduler = HeunDiscreteScheduler(self.pipeline.scheduler.config)
            elif checkpoint_settings.sampler_name == 'LMS':
                self.pipeline.scheduler = LMSDiscreteScheduler(self.pipeline.scheduler.config)

            if lora_settings is not None and len(lora_settings) > 0:
                adapter_names = []
                adapter_weights = []
                for lora_setting_item in lora_settings:
                    target_list = glob.glob(os.path.join(self.loras_dir_path, lora_setting_item.name + '.*'))
                    for target_name in target_list:
                        if os.path.splitext(target_name)[1] == '.safetensors':
                            lora_file_name = target_name
                    self.pipeline.load_lora_weights(".", weight_name=lora_file_name, adapter_name=lora_setting_item.name)
                    adapter_names.append(lora_setting_item.name)
                    adapter_weights.append(lora_setting_item.weight)
                self.pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)

        generator = torch.Generator("cuda")
        seed = generator.seed()

        prompt = prompt + ", " + checkpoint_settings.prompt
        image = self.pipeline(
            prompt,
            negative_prompt=checkpoint_settings.negative_prompt,
            guidance_scale=checkpoint_settings.cfg_scale,
            num_inference_steps=checkpoint_settings.steps,
            clip_skip=checkpoint_settings.clip_skip,
            generator=generator,
            width=checkpoint_settings.width,
            height=checkpoint_settings.height,
        ).images[0]

        if self.save_dir_path is None:
            self.save_dir_path = "sd_chat"
        dir_path = os.path.join(self.save_dir_path, "txt2img", now_str)
        os.makedirs(dir_path, exist_ok=True)
        index = len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])
        save_path = os.path.join(self.save_dir_path, "txt2img", now_str, f'{index:05}_{seed}.png')

        image.save(save_path)

        del image
        gc.collect()
        torch.cuda.empty_cache()

        self.image_results[image_id] = os.path.abspath(save_path)

    def start_txt2img(self, prompt: str, checkpoint_settings: CheckPointSettings, lora_settings: list):
        image_id = str(uuid.uuid4())
        self.image_threads[image_id] = threading.Thread(target=self.__txt2img, args=(image_id, prompt, checkpoint_settings, lora_settings))
        self.image_threads[image_id].start()
        return image_id

    async def get_result(self, image_id):
        if not image_id in self.image_threads:
            return None
        while not image_id in self.image_results:
            await asyncio.sleep(0.01)
        return self.image_results[image_id]