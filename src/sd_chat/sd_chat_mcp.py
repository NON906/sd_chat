from pathlib import Path
import threading
import json
from typing import List, Dict
import sys
from contextlib import redirect_stdout
import re
from fastmcp import FastMCP
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .sdapi_webui_client import SDAPI_WebUIClient
from .settings import CheckPointSettings, LoraSettings
from .util import get_path_settings_file
from .civitai import CivitaiAPI

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

civitai_api = CivitaiAPI()

@mcp.tool()
async def install_default_checkpoint(checkpoint_name: str) -> str:
    """Install default checkpoint model.
The targets are as follows.

- Animagine XL V3.1
- Pony Diffusion V6 XL
- Illustrious-XL
- Anything V5.0

Be sure to do this if you install them.

Args:
    checkpoint_name: Checkpoint name ('Animagine XL V3.1', 'Pony Diffusion V6 XL', 'Illustrious-XL' or 'Anything V5.0').
Return value:
    ID to check if downloading.
"""
    with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
        settings_dict = json.load(f)
    if checkpoint_name in settings_dict['checkpoints']:
        name = checkpoint_name
    else:
        for checkpoint_key, checkpoint_value in settings_dict['checkpoints'].items():
            if checkpoint_value['name'] == checkpoint_name:
                name = checkpoint_key
    
    return await civitai_api.install_model(await sd_api.get_checkpoints_dir_path(), None, None, settings_dict['checkpoints'][name]['caption'], checkpoint_name)

@mcp.tool()
async def civitai_search(query: str, page: int = 0) -> list:
    """Search Checkpoints and Loras from civitai.com.
No need to search for already on the list (get_models_list) and default checkpoint installation ('Animagine XL V3.1', 'Pony Diffusion V6 XL', 'Illustrious-XL' and 'Anything V5.0') (In that case, please use "install_default_checkpoint").

Args:
    query: Keywords to search.
    page: Index number of page to refer to.
Return value:
    The following item's list.
        model_id: Model's id used by other tools. Please be sure to include it in your reply.
        name: Model's name.
        type: 'Checkpoint' or 'LORA'.
"""
    return await civitai_api.search(query, page)

@mcp.tool()
async def civitai_url_to_id(url: str) -> dict:
    """Get model id and version id from url.

Args:
    url: URL starting with "https://civitai.com/models/".
Return value:
    model_id: Model's id.
    version_id: Version's id. (Only if this can get it)
"""
    if '?modelVersionId=' in url:
        ret_id = re.findall(r'https://civitai.com/models/(\d+)\?modelVersionId=(\d+)', url)
        return {
            'model_id': int(ret_id[0][0]),
            'version_id': int(ret_id[0][1])
        }
    else:
        ret_id = re.findall(r'https://civitai.com/models/(\d+).*', url)
        return {
            'model_id': int(ret_id[0])
        }

@mcp.tool()
async def civitai_get_versions(model_id: int) -> list:
    """List of versions of the specified model.

Args:
    model_id: Model's id.
Return value:
    The following item's list.
        version_id: Version's id.
        name: Version's name.
        base_model: Model category. Checkpoint and Lora's base_model must match.
        description: Version's description.
"""
    return await civitai_api.get_model_versions(model_id)

@mcp.tool()
async def civitai_get_model_info(version_id: int) -> dict:
    """Get model information from version_id.

Args:
    version_id: Model's version id.
Return value:
    model_name: Model name.
    model_description: Model description.
    version_name: Version name.
    version_description: Version description (if exists).
    type: The model type.
    version_base_model: Version's model category. Checkpoint and Lora's base_model must match.
"""
    return await civitai_api.get_model_info(version_id)

@mcp.tool()
async def civitai_install_model(version_id: int, caption: str, checkpoint_name: str = None, weight: float = 1.0) -> str | None:
    """Install model from version_id.

Args:
    version_id: Model's version id.
    caption: Short description of the model. Please summarize the content in "civitai_get_model_info".
    checkpoint_name: If the model is Lora, name of the base Checkpoint. Please specify one of "get_models_list".
    weight: If the model is Lora, weight to apply. Please obtain it from the contents of "civitai_get_model_info".
Return value:
    ID to check if downloading.
    If None, this ended with an error.
"""
    with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
        settings_dict = json.load(f)
    if checkpoint_name in settings_dict['checkpoints']:
        name = checkpoint_name
    else:
        for checkpoint_key, checkpoint_value in settings_dict['checkpoints'].items():
            if checkpoint_value['name'] == checkpoint_name:
                name = checkpoint_key

    return await civitai_api.install_model(await sd_api.get_checkpoints_dir_path(), await sd_api.get_loras_dir_path(), version_id, caption, name, weight)

@mcp.tool()
async def civitai_download_status(download_id: str) -> str:
    """Check if downloading.

Args:
    download_id: ID on "civitai_install_model".
Return value:
    'Downloading.' or 'Finished.' or 'Nothing.'
"""
    return civitai_api.download_status(download_id)

@mcp.tool()
async def get_models_list() -> dict:
    """List of Checkpoints and Loras used for image generation.
    
Return value:
    The following dict.
        key: Checkpoint's name.
        value: Checkpoint's summary and settings.
            name: File name.
            caption: Checkpoint's description.
            base_model: Model category. Checkpoint and Lora's base_model must match.
            installed: This checkpoint is installed.
            loras: The following dict. They are cannot be used with other Checkpoints.
                key: Lora's name.
                value: Lora's summary and settings.
                    name: File name.
                    trigger_words: Prompt required for generation with Lora. Make sure to put it in the Prompt unless it's completely different from what you want to generate.
                    caption: Lora's description.
                    base_model: Model category. Checkpoint and Lora's base_model must match.
"""
    with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
        settings_dict = json.load(f)
    ret = {}
    for checkpoint_item_name, checkpoint_setting_dict in settings_dict['checkpoints'].items():
        ret[checkpoint_item_name] = {
            'name': checkpoint_setting_dict['name'],
            'caption': checkpoint_setting_dict['caption'],
            'base_model': checkpoint_setting_dict['base_model'],
            'installed': not 'not_installed' in checkpoint_setting_dict
        }
        ret[checkpoint_item_name]['loras'] = {}
        for lora_item_name, lora_setting_dict in checkpoint_setting_dict['loras'].items():
            ret[checkpoint_item_name]['loras'][lora_item_name] = {
                'name': lora_setting_dict['name'],
                'trigger_words': lora_setting_dict['trigger_words'],
                'caption': lora_setting_dict['caption'],
                'base_model': lora_setting_dict['base_model'],
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
    Generated image's markdown tag or error message.
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
    if checkpoint_settings.not_installed:
        return f'Error: Checkpoint {checkpoint_name} is not installed. Please install it.'
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
    json_file_name = get_path_settings_file('images.json')
    if json_file_name is not None:
        with open(json_file_name, 'r', encoding="utf-8") as f:
            images_dict = json.load(f)
        if image_id in images_dict:
            return FileResponse(images_dict[image_id])
    else:
        images_dict = {}
    result_path = await sd_api.get_result(image_id)
    if result_path is None:
        raise HTTPException(status_code=404, detail="Image not found")
    images_dict[image_id] = result_path
    with open(get_path_settings_file('images.json', new_file=True), 'w', encoding="utf-8") as f:
        json.dump(images_dict, f)
    return FileResponse(result_path)

def mcp_thread_func():
    mcp.run()

def uvicorn_thread_func():
    uvicorn.run(http_app, host="0.0.0.0", port=http_port, log_level='error')

def main():
    with redirect_stdout(sys.stderr):
        run_thread = threading.Thread(target=uvicorn_thread_func)
        run_thread.start()
    mcp_thread_func()

if __name__ == "__main__":
    main()
