from fastmcp import FastMCP, Image

from src.sdapi_webui import SDAPI_WebUI
from src.settings import CheckPointSettings

mcp = FastMCP("Stable Diffusion MCP Server")

sd_api = SDAPI_WebUI()

@mcp.tool()
async def txt2img(prompt: str) -> Image:
    """Generate image with Stable Diffusion.
Prompt to specify is comma separated keywords.
If it is not in English, please translate it into English (lang:en).
For example, if you want to output "a school girl wearing a red ribbon", it would be as follows.
    1girl, school uniform, red ribbon

Args:
    prompt: The prompt to generate the image
Return value:
    Generated image
"""
    checkpoint_settings = CheckPointSettings(name='animagineXLV31_v31')
    ret_base = await sd_api.txt2img(prompt, checkpoint_settings)
    return Image(data=ret_base, format="png")

if __name__ == "__main__":
    mcp.run()
