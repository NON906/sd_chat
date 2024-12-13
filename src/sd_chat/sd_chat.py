from pathlib import Path
import http.server
import socketserver
import threading
from fastmcp import FastMCP, Image

from .sdapi_webui import SDAPI_WebUI
from .settings import CheckPointSettings

SAVE_PATH = 'sd_chat'

mcp = FastMCP("Stable Diffusion MCP Server")

sd_api = SDAPI_WebUI(save_dir_path=SAVE_PATH)

http_port = 50080
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SAVE_PATH, **kwargs)

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
    checkpoint_settings = CheckPointSettings(name='animagineXLV31_v31')
    path = await sd_api.txt2img(prompt, checkpoint_settings)
    url = f'http://localhost:{http_port}/{str(Path(path).relative_to(Path(SAVE_PATH).resolve())).replace('\\', '/')}'
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
