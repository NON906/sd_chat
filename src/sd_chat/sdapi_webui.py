import os
import webuiapi
from contextlib import redirect_stdout
import io

from .settings import CheckPointSettings

class SDAPI_WebUI:
    def __init__(self, save_dir_path: str | None = None):
        with redirect_stdout(open(os.devnull, 'w')):
            self.api = webuiapi.WebUIApi()
            self.save_dir_path = save_dir_path

    async def txt2img(self, prompt: str, checkpoint_settings: CheckPointSettings):
        with redirect_stdout(open(os.devnull, 'w')):
            if self.api.util_get_current_model() != checkpoint_settings.name:
                self.api.util_set_model(checkpoint_settings.name)
                self.api.util_wait_for_ready()
            result = await self.api.txt2img(
                prompt=prompt + ", " + checkpoint_settings.prompt,
                negative_prompt=checkpoint_settings.negative_prompt,
                cfg_scale=checkpoint_settings.cfg_scale,
                sampler_name=checkpoint_settings.sampler_name,
                steps=checkpoint_settings.steps,
                width=checkpoint_settings.width,
                height=checkpoint_settings.height,
                enable_hr=checkpoint_settings.enable_hr,
                hr_upscaler=checkpoint_settings.hr_upscaler,
                hr_second_pass_steps=checkpoint_settings.hr_second_pass_steps,
                hr_resize_x=checkpoint_settings.hr_resize_x,
                hr_resize_y=checkpoint_settings.hr_resize_y,
                denoising_strength=checkpoint_settings.denoising_strength,
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