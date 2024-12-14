from pathlib import Path
import http.server
import socketserver
import threading
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
async def txt2img(prompt: str, lora_names: List[str] = []) -> str:
    """Generate image with Stable Diffusion.
Prompt to specify is comma separated keywords.
If it is not in English, please translate it into English (lang:en).
For example, if you want to output "a school girl wearing a red ribbon", it would be as follows.
    1girl, school uniform, red ribbon

Args:
    prompt: The prompt to generate the image.
    lora_names: List of Lora names to use. Leave blank if not used.
Return value:
    Generated image markdown tag.
"""
    with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
        settings_dict = json.load(f)
    checkpoint_settings = CheckPointSettings(**list(settings_dict['checkpoints'].values())[0])
    lora_settings = []
    for lora_name in lora_names:
        for lora_item_name, lora_setting_dict in settings_dict['loras'].items():
            if lora_item_name == lora_name or lora_setting_dict['name'] == lora_name:
                lora_settings.append(LoraSettings(**lora_setting_dict))
    path = await sd_api.txt2img(prompt, checkpoint_settings, lora_settings)
    url = f'http://localhost:{http_port}/{str(Path(path).relative_to(Path(save_path).resolve())).replace('\\', '/')}'
    return f'![Generation Result]({url})'

@mcp.tool()
async def get_loras_list() -> Dict[str, LoraSettings]:
    """List of Lora used for image generation.
    
Return value:
    The following dict.
        key: Lora's name.
        value: Lora's summary and settings.
            name: File name.
            trigger_words: Prompt required for generation with Lora. Make sure to put it in the Prompt unless it's completely different from what you want to generate.
            weight: Lora's recommended weight.
            caption: Lora's description.     
"""
    with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
        settings_dict = json.load(f)
    ret = {}
    for lora_item_name, lora_setting_dict in settings_dict['loras'].items():
        ret[lora_item_name] = LoraSettings(**lora_setting_dict)
    return ret

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
    http_thread = threading.Thread(target=http_server_start)
    http_thread.start()

    mcp.run()

if __name__ == "__main__":
    main()
