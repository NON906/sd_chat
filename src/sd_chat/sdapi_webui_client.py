import os
import sys
import webuiapi
from contextlib import redirect_stdout
from PIL import PngImagePlugin

from .settings import CheckPointSettings, LoraSettings

class SDAPI_WebUIClient:
    def __init__(self, save_dir_path: str | None = None):
        with redirect_stdout(sys.stderr):
            self.api = webuiapi.WebUIApi()
            self.save_dir_path = save_dir_path

    async def txt2img(self, prompt: str, checkpoint_settings: CheckPointSettings, lora_settings: list):
        with redirect_stdout(sys.stderr):
            prev_options = self.api.get_options()
            changed_options = {}
            if checkpoint_settings.clip_skip != prev_options['CLIP_stop_at_last_layers']:
                changed_options['CLIP_stop_at_last_layers'] = prev_options['CLIP_stop_at_last_layers']
                self.api.set_options({'CLIP_stop_at_last_layers': checkpoint_settings.clip_skip})
            if not checkpoint_settings.name in prev_options['sd_model_checkpoint']:
                changed_options['sd_model_checkpoint'] = prev_options['sd_model_checkpoint']
                self.api.util_set_model(checkpoint_settings.name)
            if len(changed_options.items()) > 0:
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

            if len(changed_options.items()) > 0:
                self.api.set_options(changed_options)

            if self.save_dir_path is None:
                self.save_dir_path = "sd_chat"
            os.makedirs(os.path.join(self.save_dir_path, "txt2img"), exist_ok=True)
            index = 0
            save_path = os.path.join(self.save_dir_path, "txt2img", f'{index:08}.png')
            while os.path.exists(save_path):
                index += 1
                save_path = os.path.join(self.save_dir_path, "txt2img", f'{index:08}.png')

            metadata = PngImagePlugin.PngInfo()
            metadata.add_text('parameters', result.info['infotexts'][0])

            result.image.save(save_path, pnginfo=metadata)

            return os.path.abspath(save_path)