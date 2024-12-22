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

    host = '127.0.0.1'
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

async def image_api_dropdown_func(image_api_name):
    global settings_dict
    settings_dict['target_api'] = image_api_name

    if image_api_name == 'diffusers':
        image_api_name = 'Diffusers'
    elif image_api_name == 'webui_client':
        image_api_name = 'sdwebuiapi'
    
    ret = [
        gr.update(visible=image_api_name == 'Diffusers'),
        gr.update(visible=image_api_name == 'Diffusers'),
        gr.update(visible=image_api_name == 'sdwebuiapi'),
        gr.update(visible=image_api_name == 'sdwebuiapi'),
    ]
    return ret

async def image_api_webui_clients_append():
    global settings_dict
    if not 'apis' in settings_dict:
        settings_dict['apis'] = {}
    if not 'webui_client' in settings_dict['apis']:
        settings_dict['apis']['webui_client'] = {}

async def ui_init_models(model):
    if model in settings_dict['checkpoints']:
        lora_choices = []
        for loop_item in settings_dict['checkpoints'][model]['loras'].keys():
            lora_choices.append(loop_item)
        ret = [
            settings_dict['checkpoints'][model]['name'] if 'name' in settings_dict['checkpoints'][model] else '',
            settings_dict['checkpoints'][model]['caption'] if 'caption' in settings_dict['checkpoints'][model] else '',
            settings_dict['checkpoints'][model]['prompt'] if 'prompt' in settings_dict['checkpoints'][model] else '',
            settings_dict['checkpoints'][model]['negative_prompt'] if 'negative_prompt' in settings_dict['checkpoints'][model] else '',
            settings_dict['checkpoints'][model]['cfg_scale'] if 'cfg_scale' in settings_dict['checkpoints'][model] else 7.0,
            settings_dict['checkpoints'][model]['sampler_name'] if 'sampler_name' in settings_dict['checkpoints'][model] else 'Euler a',
            settings_dict['checkpoints'][model]['steps'] if 'steps' in settings_dict['checkpoints'][model] else 20,
            settings_dict['checkpoints'][model]['width'] if 'width' in settings_dict['checkpoints'][model] else 512,
            settings_dict['checkpoints'][model]['height'] if 'height' in settings_dict['checkpoints'][model] else 512,
            settings_dict['checkpoints'][model]['clip_skip'] if 'clip_skip' in settings_dict['checkpoints'][model] else 1,
            settings_dict['checkpoints'][model]['base_model'] if 'base_model' in settings_dict['checkpoints'][model] else '',
            gr.update(choices=lora_choices, value=''),
            '',
            [],
            1.0,
            ''
        ]
        return ret
    ret = []
    for _ in range(16):
        ret.append(gr.update())
    return ret

async def models_append(model):
    global settings_dict
    if not 'checkpoints' in settings_dict:
        settings_dict['checkpoints'] = {}
    if not model in settings_dict['checkpoints']:
        settings_dict['checkpoints'][model] = {}

async def ui_init_loras(model, lora):
    if lora in settings_dict['checkpoints'][model]['loras']:
        ret = [
            settings_dict['checkpoints'][model]['loras'][lora]['name'] if 'name' in settings_dict['checkpoints'][model]['loras'][lora] else '',
            settings_dict['checkpoints'][model]['loras'][lora]['trigger_words'] if 'trigger_words' in settings_dict['checkpoints'][model]['loras'][lora] else [],
            settings_dict['checkpoints'][model]['loras'][lora]['weight'] if 'weight' in settings_dict['checkpoints'][model]['loras'][lora] else 1.0,
            settings_dict['checkpoints'][model]['loras'][lora]['caption'] if 'caption' in settings_dict['checkpoints'][model]['loras'][lora] else '',
        ]
        return ret
    ret = []
    for _ in range(4):
        ret.append(gr.update())
    return ret

async def loras_append(model, lora):
    global settings_dict
    if 'loras' in settings_dict['checkpoints'][model]:
        settings_dict['checkpoints'][model]['loras'] = {}
    if lora in settings_dict['checkpoints'][model]['loras']:
        settings_dict['checkpoints'][model]['loras'][lora] = {}

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
                    file_load_btn = gr.Button(value='Load')
                gr.Markdown(value='## Image Generation API')
                image_api_dropdown = gr.Dropdown(choices=['Diffusers', 'sdwebuiapi'], label='API', interactive=True)
                with gr.Row():
                    image_api_checkpoints_path_textbox = gr.Textbox(label='Checkpoints Path', visible=False)
                    image_api_loras_path_textbox = gr.Textbox(label='Loras Path', visible=False)
                    image_api_host_textbox = gr.Textbox(label='Host Name', value='127.0.0.1', visible=False)
                    image_api_port_numberbox = gr.Number(label='Port', value=7860, visible=False, interactive=True)
                gr.Markdown(value='## Civitai API Key')
                civitai_api_key_textbox = gr.Textbox(label='API Key')
                civitai_disable_tools_checkbox = gr.Checkbox(label='Disable some civitai tools.', value=False) # (Restart required)
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
                with gr.Row():
                    models_cfg_scale_numberbox = gr.Number(label='Cfg Scale', value=7.0, interactive=True)
                    models_sampler_name_dropdown = gr.Dropdown(choices=['DPM++ 2M', 'DPM++ SDE', 'DPM2', 'DPM2 a', 'Euler', 'Euler a', 'Heun', 'LMS'], value='Euler a', label='Sampler', allow_custom_value=True, interactive=True)
                    models_steps_numberbox = gr.Number(label='Steps', value=20, interactive=True)
                    models_clip_skip_numberbox = gr.Number(label='Clip Skip', value=1, interactive=True)
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

        image_api_dropdown.input(
            image_api_dropdown_func, inputs=image_api_dropdown, outputs=[
                image_api_checkpoints_path_textbox,
                image_api_loras_path_textbox,
                image_api_host_textbox,
                image_api_port_numberbox,
            ]
        )

        def func(x):
            settings_dict['checkpoints_path'] = x
        image_api_checkpoints_path_textbox.input(
            func, inputs=image_api_checkpoints_path_textbox
        )
        def func(x):
            settings_dict['lora_path'] = x
        image_api_loras_path_textbox.input(
            func, inputs=image_api_loras_path_textbox
        )
        def func(x):
            settings_dict['apis']['webui_client']['host'] = x
        image_api_host_textbox.input(
            image_api_webui_clients_append
        ).then(
            func, inputs=image_api_host_textbox
        )
        def func(x):
            settings_dict['apis']['webui_client']['port'] = x
        image_api_port_numberbox.input(
            image_api_webui_clients_append
        ).then(
            func, inputs=image_api_port_numberbox
        )

        def func(x):
            settings_dict['civitai_api_key'] = x
        civitai_api_key_textbox.input(
            func, inputs=civitai_api_key_textbox
        )

        def func(x):
            settings_dict['disable_civitai_tools'] = x
        civitai_disable_tools_checkbox.input(
            func, inputs=civitai_disable_tools_checkbox
        )

        models_dropdown.input(
            ui_init_models, inputs=models_dropdown, outputs=[
                models_name_textbox,
                models_caption_textbox,
                models_prompt_textbox,
                models_negative_prompt_textbox,
                models_cfg_scale_numberbox,
                models_sampler_name_dropdown,
                models_steps_numberbox,
                models_width_numberbox,
                models_height_numberbox,
                models_clip_skip_numberbox,
                models_base_model_dropdown,
                loras_dropdown,
                loras_name_textbox,
                loras_trigger_words_dropdown,
                loras_weight_numberbox,
                loras_caption_textbox,
            ]
        )

        def func(x, y):
            settings_dict['checkpoints'][x]['name'] = y
        models_name_textbox.input(
            models_append, inputs=models_dropdown
        ).then(
            func, inputs=[models_dropdown, models_name_textbox]
        )
        def func(x, y):
            settings_dict['checkpoints'][x]['caption'] = y
        models_caption_textbox.input(
            models_append, inputs=models_dropdown
        ).then(
            func, inputs=[models_dropdown, models_caption_textbox]
        )
        def func(x, y):
            settings_dict['checkpoints'][x]['prompt'] = y
        models_prompt_textbox.input(
            models_append, inputs=models_dropdown
        ).then(
            func, inputs=[models_dropdown, models_prompt_textbox]
        )
        def func(x, y):
            settings_dict['checkpoints'][x]['negative_prompt'] = y
        models_negative_prompt_textbox.input(
            models_append, inputs=models_dropdown
        ).then(
            func, inputs=[models_dropdown, models_negative_prompt_textbox]
        )
        def func(x, y):
            settings_dict['checkpoints'][x]['base_model'] = y
        models_base_model_dropdown.input(
            models_append, inputs=models_dropdown
        ).then(
            func, inputs=[models_dropdown, models_base_model_dropdown]
        )
        def func(x, y):
            settings_dict['checkpoints'][x]['width'] = y
        models_width_numberbox.input(
            models_append, inputs=models_dropdown
        ).then(
            func, inputs=[models_dropdown, models_width_numberbox]
        )
        def func(x, y):
            settings_dict['checkpoints'][x]['height'] = y
        models_height_numberbox.input(
            models_append, inputs=models_dropdown
        ).then(
            func, inputs=[models_dropdown, models_height_numberbox]
        )
        def func(x, y):
            settings_dict['checkpoints'][x]['cfg_scale'] = y
        models_height_numberbox.input(
            models_append, inputs=models_dropdown
        ).then(
            func, inputs=[models_dropdown, models_cfg_scale_numberbox]
        )
        def func(x, y):
            settings_dict['checkpoints'][x]['sampler_name'] = y
        models_sampler_name_dropdown.input(
            models_append, inputs=models_dropdown
        ).then(
            func, inputs=[models_dropdown, models_sampler_name_dropdown]
        )
        def func(x, y):
            settings_dict['checkpoints'][x]['steps'] = y
        models_steps_numberbox.input(
            models_append, inputs=models_dropdown
        ).then(
            func, inputs=[models_dropdown, models_steps_numberbox]
        )
        def func(x, y):
            settings_dict['checkpoints'][x]['clip_skip'] = y
        models_clip_skip_numberbox.input(
            models_append, inputs=models_dropdown
        ).then(
            func, inputs=[models_dropdown, models_clip_skip_numberbox]
        )

        loras_dropdown.input(
            ui_init_loras, inputs=[models_dropdown, loras_dropdown], outputs=[
                loras_name_textbox,
                loras_trigger_words_dropdown,
                loras_weight_numberbox,
                loras_caption_textbox,
            ]
        )

        def func(x, y, z):
            settings_dict['checkpoints'][x]['loras'][y]['name'] = z
        loras_name_textbox.input(
            models_append, inputs=models_dropdown
        ).then(
            loras_append, inputs=[models_dropdown, loras_dropdown]
        ).then(
            func, inputs=[models_dropdown, loras_dropdown, loras_name_textbox]
        )
        def func(x, y, z):
            settings_dict['checkpoints'][x]['loras'][y]['trigger_words'] = z
        loras_trigger_words_dropdown.input(
            models_append, inputs=models_dropdown
        ).then(
            loras_append, inputs=[models_dropdown, loras_dropdown]
        ).then(
            func, inputs=[models_dropdown, loras_dropdown, loras_trigger_words_dropdown]
        )
        def func(x, y, z):
            settings_dict['checkpoints'][x]['loras'][y]['weight'] = z
        loras_weight_numberbox.input(
            models_append, inputs=models_dropdown
        ).then(
            loras_append, inputs=[models_dropdown, loras_dropdown]
        ).then(
            func, inputs=[models_dropdown, loras_dropdown, loras_weight_numberbox]
        )
        def func(x, y, z):
            settings_dict['checkpoints'][x]['loras'][y]['caption'] = z
        loras_caption_textbox.input(
            models_append, inputs=models_dropdown
        ).then(
            loras_append, inputs=[models_dropdown, loras_dropdown]
        ).then(
            func, inputs=[models_dropdown, loras_dropdown, loras_caption_textbox]
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