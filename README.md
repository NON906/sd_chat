# sd_chat

（[日本語版はこちら](README_ja.md)）

This tool enables executing txt2img generation via StableDiffusion, and installing Checkpoints and Loras from [Civitai](https://civitai.com/) directly through chats with an LLM.  
Currently, it can only run in the [MCP Server](https://modelcontextprotocol.io/introduction) format.  
It has been tested with the [Claude Desktop App](https://claude.ai/download), [Model Context Protocol CLI](https://github.com/chrishayuk/mcp-cli), and [MCP-Bridge](https://github.com/SecretiveShell/MCP-Bridge).

## Installation

1. [Install uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation).
2. Clone this repository using ``git clone``.
3. Move to the repository directory (``cd sd_chat``) and ``run uv sync --all-extras``.
4. Open the settings UI by running ``uv run sd-chat-ui``.
5. Configure the "Image Generation API" and "Civitai API Key", then click the "Save" button.
6. A JSON template for the "MCP Server Template" will be displayed. Copy this JSON and add it to the configuration file for your MCP client  
(e.g., for the Windows version of the Claude Desktop App, add it to ``C:\Users\xxx\AppData\Roaming\Claude\claude_desktop_config.json``).

## Usage Examples

- Do you have a Lora for xxx?
- Please install the Lora for xxx.
- Generate an image of a girl using xxx(model name).

etc.