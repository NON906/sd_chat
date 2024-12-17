from pathlib import Path
import threading
import json
from typing import List, Dict
import sys
from contextlib import redirect_stdout
from fastmcp import FastMCP
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

from .sdapi_webui_client import SDAPI_WebUIClient
from .settings import CheckPointSettings, LoraSettings
from .util import get_path_settings_file

with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
    settings_dict = json.load(f)

save_path = settings_dict['save_path']

mcp = FastMCP("Stable Diffusion MCP Server")

if settings_dict['target_api'] == 'webui_client':
    if 'apis' in settings_dict and 'webui_client' in settings_dict['apis']:
        sd_api = SDAPI_WebUIClient(save_dir_path=save_path, settings=settings_dict['apis']['webui_client'])
    else:
        sd_api = SDAPI_WebUIClient(save_dir_path=save_path)

http_port = 50080

http_app = FastAPI()

@mcp.tool()
async def get_models_list() -> dict:
    """List of Checkpoints and Loras used for image generation.
    
Return value:
    The following dict.
        key: Checkpoint's name.
        value: Checkpoint's summary and settings.
            name: File name.
            caption: Checkpoint's description.
            loras: The following dict. They are cannot be used with other Checkpoints.
                key: Lora's name.
                value: Lora's summary and settings.
                    name: File name.
                    trigger_words: Prompt required for generation with Lora. Make sure to put it in the Prompt unless it's completely different from what you want to generate.
                    caption: Lora's description.
"""
    with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
        settings_dict = json.load(f)
    ret = {}
    for checkpoint_item_name, checkpoint_setting_dict in settings_dict['checkpoints'].items():
        ret[checkpoint_item_name] = {
            'name': checkpoint_setting_dict['name'],
            'caption': checkpoint_setting_dict['caption'],
        }
        ret[checkpoint_item_name]['loras'] = {}
        for lora_item_name, lora_setting_dict in checkpoint_setting_dict['loras'].items():
            ret[checkpoint_item_name]['loras'][lora_item_name] = {
                'name': lora_setting_dict['name'],
                'trigger_words': lora_setting_dict['trigger_words'],
                'caption': lora_setting_dict['caption'],
            }
    return ret

@mcp.tool()
async def txt2img(prompt: str, checkpoint_name: str, lora_names: List[str] = []) -> str:
    """Generate image with Stable Diffusion.
Prompt to specify is comma separated keywords.
If it is not in English, please translate it into English (lang:en).
Don't use escape characters and multibyte characters.
For example, if you want to output "a school girl wearing a red ribbon", it would be as follows.
    1girl, school uniform, red ribbon
You can use Lora as an additional option.
In that case, you need to set it in "lora_names" and then put trigger_words in "prompt".

Args:
    prompt: The prompt to generate the image. Don't use escape characters and multibyte characters.
    checkpoint_name: Checkpoint name to use.
    lora_names: List of Lora names to use. Leave blank if not used.
Return value:
    Generated image's markdown tag.
"""
    if type(lora_names) is str:
        lora_names = [lora_names, ]
    elif type(lora_names) is not list:
        lora_names = []
    with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
        settings_dict = json.load(f)
    if checkpoint_name in settings_dict['checkpoints']:
        checkpoint_settings = CheckPointSettings(**settings_dict['checkpoints'][checkpoint_name])
    else:
        for checkpoint_key, checkpoint_value in settings_dict['checkpoints'].items():
            if checkpoint_value['name'] == checkpoint_name:
                checkpoint_settings = CheckPointSettings(**checkpoint_value)
    lora_settings = []
    for lora_name in lora_names:
        for lora_item_name, lora_setting_dict in checkpoint_settings.loras.items():
            if lora_item_name == lora_name or lora_setting_dict.name == lora_name:
                lora_settings.append(lora_setting_dict)
    image_id = sd_api.start_txt2img(prompt, checkpoint_settings, lora_settings)
    url = f'http://localhost:{http_port}/get_result/{image_id}'
    return f'![Generation Result]({url})'

@http_app.get("/get_result/{image_id}")
async def get_result(image_id: str):
    result_path = await sd_api.get_result(image_id)
    return FileResponse(result_path)

def uvicorn_thread_func():
    uvicorn.run(http_app, host="0.0.0.0", port=http_port)

def main():
    with redirect_stdout(sys.stderr):
        uvicorn_thread = threading.Thread(target=uvicorn_thread_func)
        uvicorn_thread.start()
    mcp.run()

if __name__ == "__main__":
    main()
