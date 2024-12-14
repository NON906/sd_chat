import os
import webuiapi
from contextlib import redirect_stdout
import io

from .settings import CheckPointSettings, LoraSettings

class SDAPI_WebUIClient:
    def __init__(self, save_dir_path: str | None = None):
        with redirect_stdout(open(os.devnull, 'w')):
            self.api = webuiapi.WebUIApi()
            self.save_dir_path = save_dir_path

    async def txt2img(self, prompt: str, checkpoint_settings: CheckPointSettings, lora_settings: list):
        with redirect_stdout(open(os.devnull, 'w')):
            if checkpoint_settings.name in self.api.util_get_current_model():
                self.api.util_set_model(checkpoint_settings.name)
                self.api.util_wait_for_ready()
            lora_prompt = ''
            for lora_settings_item in lora_settings:
                lora_prompt += f', <lora:{lora_settings_item.name}:{lora_settings_item.weight}>'
            result = await self.api.txt2img(
                prompt=prompt + ", " + checkpoint_settings.prompt + lora_prompt,
                negative_prompt=checkpoint_settings.negative_prompt,
                cfg_scale=checkpoint_settings.cfg_scale,
                sampler_name=checkpoint_settings.sampler_name,
                steps=checkpoint_settings.steps,
                width=checkpoint_settings.width,
                height=checkpoint_settings.height,
                use_async=True,
            )
            if self.save_dir_path is None:
                self.save_dir_path = "sd_chat"
            os.makedirs(os.path.join(self.save_dir_path, "txt2img"), exist_ok=True)
            index = 0
            save_path = os.path.join(self.save_dir_path, "txt2img", f'{index:08}.png')
            while os.path.exists(save_path):
                index += 1
                save_path = os.path.join(self.save_dir_path, "txt2img", f'{index:08}.png')
            result.image.save(save_path)
            return os.path.abspath(save_path)