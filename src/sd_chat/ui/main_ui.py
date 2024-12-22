import os
import json
import logging
import aiohttp
import gradio as gr

from ..util import get_path_settings_file

settings_dict = {}

async def reload_settings(file_path):
    global settings_dict
    with open(file_path, 'r', encoding="utf-8") as f:
        settings_dict = json.load(f)

async def ui_init():
    image_api_name = settings_dict['target_api']
    if settings_dict['target_api'] == 'diffusers':
        image_api_name = 'Diffusers'
    elif settings_dict['target_api'] == 'webui_client':
        image_api_name = 'sdwebuiapi'

    host = ''
    port = 7860
    if 'apis' in settings_dict and 'webui_client' in settings_dict['apis']:
        if 'host' in settings_dict['apis']['webui_client']:
            host = settings_dict['apis']['webui_client']['host']
        if 'port' in settings_dict['apis']['webui_client']:
            port = settings_dict['apis']['webui_client']['port']

    chat_api = settings_dict['chat_api'] if 'chat_api' in settings_dict else 'OpenAI API'

    models_list = []
    for model in settings_dict['checkpoints'].keys():
        models_list.append(model)

    ret = [
        get_path_settings_file('settings.json', new_file=True),
        image_api_name,
        gr.update(visible=image_api_name == 'Diffusers', value=settings_dict['checkpoints_path'] if 'checkpoints_path' in settings_dict else ''),
        gr.update(visible=image_api_name == 'Diffusers', value=settings_dict['lora_path'] if 'lora_path' in settings_dict else ''),
        gr.update(visible=image_api_name == 'sdwebuiapi', value=host),
        gr.update(visible=image_api_name == 'sdwebuiapi', value=port),
        settings_dict['civitai_api_key'] if 'civitai_api_key' in settings_dict else '',
        chat_api,
        gr.update(visible=chat_api == 'Other (OpenAI API compatible)', value=settings_dict['chat_api_url'] if 'chat_api_url' in settings_dict else ''),
        settings_dict['chat_api_key'] if 'chat_api_key' in settings_dict else '',
        gr.update(choices=models_list),
    ]
    return ret

async def ui_init_chat_model_names(chat_api, chat_api_url, chat_api_api_key):
    if chat_api_api_key is None or chat_api_api_key == '':
        return gr.update(choices=[], value=settings_dict['chat_api_model'] if 'chat_api_model' in settings_dict else '')
    if chat_api == 'OpenAI API':
        chat_api_url = 'https://api.openai.com/v1'
    headers = {"Authorization": f"Bearer {chat_api_api_key}"}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(chat_api_url + '/models') as response:
            response_dict = await response.json()
    ret_list = []
    for model_dict in response_dict['data']:
        ret_list.append(model_dict['id'])
    return gr.update(choices=ret_list, value=settings_dict['chat_api_model'] if 'chat_api_model' in settings_dict else '')

async def save_settings(file_path):
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(settings_dict, f, indent=2)

def main_ui(platform='standalone'):
    global settings_dict
    with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
        settings_dict = json.load(f)

    with gr.Blocks() as runner_interface:
        with gr.Tabs() as main_tabs:
            with gr.TabItem('Settings', id='Settings') as main_settings_tab:
                gr.Markdown(value='## File')
                file_path_textbox = gr.Textbox(label='Path', value='')
                with gr.Row():
                    file_save_btn = gr.Button(value='Save', variant='primary')
                    file_load_btn = gr.Button(value='Load', variant='primary')
                gr.Markdown(value='## Image Generation API')
                image_api_dropdown = gr.Dropdown(choices=['Diffusers', 'sdwebuiapi'], label='API', interactive=True)
                with gr.Row():
                    image_api_checkpoints_path_textbox = gr.Textbox(label='Checkpoints Path', visible=False)
                    image_api_loras_path_textbox = gr.Textbox(label='Loras Path', visible=False)
                    image_api_host_textbox = gr.Textbox(label='Host Name', value='127.0.0.1', visible=False)
                    image_api_port_numberbox = gr.Number(label='Port', value=7860, visible=False, interactive=True)
                gr.Markdown(value='## Civitai API Key')
                civitai_api_key_textbox = gr.Textbox(label='API Key')
                with gr.Group(visible=False):
                    gr.Markdown(value='## Chat API')
                    chat_api_dropdown = gr.Dropdown(choices=['OpenAI API', 'Other (OpenAI API compatible)'], label='API', interactive=True)
                    with gr.Row():
                        chat_api_url_textbox = gr.Textbox(label='URL', visible=False)
                        chat_api_api_key_textbox = gr.Textbox(label='API Key')
                        chat_api_model_name_dropdown = gr.Dropdown(label='Model Name', choices=[], allow_custom_value=True, interactive=True)
                gr.Markdown(value='## Image Generation Models')
                with gr.Group(visible=False):
                    download_models_btn = gr.Button(value='Download Default Models')
                    models_auto_settings_btn = gr.Button(value='Auto Settings (NOTE: Use "Chat API")')
                gr.Markdown(value='### Checkpoints')
                models_dropdown = gr.Dropdown(choices=[], value='', label='Models', allow_custom_value=True, interactive=True)
                models_name_textbox = gr.Textbox(label='Name (File name without extension)')
                models_caption_textbox = gr.Textbox(label='Caption (Description)')
                models_prompt_textbox = gr.Textbox(label='Prompt')
                models_negative_prompt_textbox = gr.Textbox(label='Negative Prompt')
                models_base_model_dropdown = gr.Dropdown(choices=['SDXL 1.0', 'Pony', 'Illustrious', 'SD 1.5'], label='Base Model', allow_custom_value=True, interactive=True)
                with gr.Row():
                    models_width_numberbox = gr.Number(label='width', value=512, interactive=True)
                    models_height_numberbox = gr.Number(label='height', value=512, interactive=True)
                gr.Markdown(value='### Loras')
                loras_dropdown = gr.Dropdown(choices=[], value='', label='Loras', allow_custom_value=True, interactive=True)
                loras_name_textbox = gr.Textbox(label='Name (File name without extension)')
                loras_trigger_words_dropdown = gr.Dropdown(choices=[], value=[], label='Trigger words', multiselect=True, allow_custom_value=True, interactive=True)
                loras_weight_numberbox = gr.Number(label='Weight', value=1.0, interactive=True)
                loras_caption_textbox = gr.Textbox(label='Caption (Description)')

        runner_interface.load(ui_init, outputs=[
            file_path_textbox,
            image_api_dropdown,
            image_api_checkpoints_path_textbox,
            image_api_loras_path_textbox,
            image_api_host_textbox,
            image_api_port_numberbox,
            civitai_api_key_textbox,
            chat_api_dropdown,
            chat_api_url_textbox,
            chat_api_api_key_textbox,
            models_dropdown,
        ]).then(
            ui_init_chat_model_names, inputs=[
                chat_api_dropdown,
                chat_api_url_textbox,
                chat_api_api_key_textbox,
            ],
            outputs=chat_api_model_name_dropdown
        )

        file_load_btn.click(
            reload_settings, inputs=file_path_textbox
        ).then(ui_init, outputs=[
            file_path_textbox,
            image_api_dropdown,
            image_api_checkpoints_path_textbox,
            image_api_loras_path_textbox,
            image_api_host_textbox,
            image_api_port_numberbox,
            civitai_api_key_textbox,
            chat_api_dropdown,
            chat_api_url_textbox,
            chat_api_api_key_textbox,
            models_dropdown,
        ]).then(
            ui_init_chat_model_names, inputs=[
                chat_api_dropdown,
                chat_api_url_textbox,
                chat_api_api_key_textbox,
            ],
            outputs=chat_api_model_name_dropdown
        )

        file_save_btn.click(
            save_settings, inputs=file_path_textbox
        )

    return runner_interface

def main():
    runner_interface = main_ui()
    runner_interface.queue()
    runner_interface.launch(server_port=50081)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.WARNING)

if __name__ == '__main__':
    main()