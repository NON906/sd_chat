from pathlib import Path
import http.server
import socketserver
import threading
import json
from fastmcp import FastMCP, Image

from .sdapi_webui_client import SDAPI_WebUIClient
from .settings import CheckPointSettings
from .util import get_path_settings_file

with open(get_path_settings_file('settings.json'), 'r') as f:
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
async def txt2img(prompt: str) -> str:
    """Generate image with Stable Diffusion.
Prompt to specify is comma separated keywords.
If it is not in English, please translate it into English (lang:en).
For example, if you want to output "a school girl wearing a red ribbon", it would be as follows.
    1girl, school uniform, red ribbon

Args:
    prompt: The prompt to generate the image
Return value:
    Generated image markdown tag
"""
    with open(get_path_settings_file('settings.json'), 'r') as f:
        settings_dict = json.load(f)
    checkpoint_settings = CheckPointSettings(**list(settings_dict['checkpoints'].values())[0])
    path = await sd_api.txt2img(prompt, checkpoint_settings)
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
    http_thread = threading.Thread(target=http_server_start)
    http_thread.start()

    mcp.run()

if __name__ == "__main__":
    main()
