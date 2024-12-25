# sd_chat

StableDiffusionのtxt2imgによる生成や、[Civitai](https://civitai.com/)からのCheckpoint・Loraのインストールを、LLMとのチャットから実行できるようにするためのものです。  
現在、[MCPサーバー](https://modelcontextprotocol.io/introduction)形式でのみ実行することができます。  
[Claude Desktop App](https://claude.ai/download)や[Model Context Protocol CLI](https://github.com/chrishayuk/mcp-cli)、[MCP-Bridge](https://github.com/SecretiveShell/MCP-Bridge)で動作確認しています。  

## インストール方法

1. [uvをインストール](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)してください。
2. このリポジトリを``git clone``してください。
3. リポジトリに移動（``cd sd_chat``）し、``uv sync --all-extras``を実行してください。
4. ``uv run sd-chat-ui``で設定画面を開いてください。
5. 「Image Generation API」と「Civitai API Key」を設定し、「Save」ボタンを押してください。
6. 「MCP Server Template」にインストール用のjsonが表示されるため、これを設定用のjsonファイル  （Windows版のClaude Desktop Appなら``C:\Users\xxx\AppData\Roaming\Claude\claude_desktop_config.json``）  
に記載してください。

## 使用方法

- ○○のLoraはありますか？
- ○○のLoraをインストールしてください。
- （モデル名）で女の子の画像を生成してください。

etc.