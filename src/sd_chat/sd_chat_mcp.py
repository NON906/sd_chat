from pathlib import Path
import http.server
import socketserver
import _thread as thread
import json
from typing import List, Dict
from fastmcp import FastMCP, Image

from .sdapi_webui_client import SDAPI_WebUIClient
from .settings import CheckPointSettings, LoraSettings
from .util import get_path_settings_file

with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
    settings_dict = json.load(f)

save_path = settings_dict['save_path']

mcp = FastMCP("Stable Diffusion MCP Server")

if settings_dict['target_api'] == 'webui_client':
    sd_api = SDAPI_WebUIClient(save_dir_path=save_path)

http_port = 50080
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=save_path, **kwargs)

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
For example, if you want to output "a school girl wearing a red ribbon", it would be as follows.
    1girl, school uniform, red ribbon

Args:
    prompt: The prompt to generate the image.
    checkpoint_name: Checkpoint name to use.
    lora_names: List of Lora names to use. Leave blank if not used.
Return value:
    Generated image markdown tag.
"""
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
    path = await sd_api.txt2img(prompt, checkpoint_settings, lora_settings)
    url = f'http://localhost:{http_port}/{str(Path(path).relative_to(Path(save_path).resolve())).replace('\\', '/')}'
    return f'![Generation Result]({url})'

def http_server_start():
    global http_port
    is_oserror = True
    while is_oserror:
        is_oserror = False
        try:
            with socketserver.TCPServer(("", http_port), Handler) as httpd:
                httpd.serve_forever()
        except OSError:
            is_oserror = True
            http_port += 1

def main():
    thread.start_new_thread(http_server_start, ())

    mcp.run()

if __name__ == "__main__":
    main()
