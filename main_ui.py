import gradio as gr

def main_ui(platform='standalone'):
    with gr.Blocks() as block_interface:
        with gr.Tabs() as main_tabs:
            with gr.TabItem('Settings', id='Settings') as main_settings_tab:
                gr.Markdown(value='## Image Generation API')
                image_api_dropdown = gr.Dropdown(choices=['Diffusers', 'sdwebuiapi'], label='API', interactive=True)
                with gr.Row():
                    image_api_checkpoints_path_textbox = gr.Textbox(label='Checkpoints Path', visible=False)
                    image_api_loras_path_textbox = gr.Textbox(label='Loras Path', visible=False)
                    image_api_host_textbox = gr.Textbox(label='Host Name', value='127.0.0.1', visible=False)
                    image_api_port_numberbox = gr.Number(label='Port', value=7860, visible=False, interactive=True)
                gr.Markdown(value='## Civitai API Key')
                civitai_api_key_textbox = gr.Textbox(label='API Key')
                gr.Markdown(value='## Chat API')
                chat_api_dropdown = gr.Dropdown(choices=['OpenAI API', 'Other (OpenAI API compatible)'], label='API', interactive=True)
                with gr.Row():
                    chat_api_url_textbox = gr.Textbox(label='URL', visible=False)
                    chat_api_api_key_textbox = gr.Textbox(label='API Key', visible=False)
                    chat_api_model_name_dropdown = gr.Dropdown(label='Model Name', choices=[], allow_custom_value=True, interactive=True)
                gr.Markdown(value='## Models')
                download_models_btn = gr.Button(value='Download Default Models')
                models_auto_settings_btn = gr.Button(value='Auto Settings (NOTE: Use "Chat API")')
                gr.Markdown(value='### Checkpoints')
                models_dropdown = gr.Dropdown(choices=[], label='Models', allow_custom_value=True, interactive=True)
                models_name_textbox = gr.Textbox(label='Name (File name without extension)')
                models_caption_textbox = gr.Textbox(label='Caption (Description)')
                models_prompt_textbox = gr.Textbox(label='Prompt')
                models_negative_prompt_textbox = gr.Textbox(label='Negative Prompt')
                models_base_model_dropdown = gr.Dropdown(choices=['SDXL 1.0', 'Pony', 'Illustrious', 'SD 1.5'], label='Base Model', allow_custom_value=True, interactive=True)
                with gr.Row():
                    models_width_numberbox = gr.Number(label='width', value=512, interactive=True)
                    models_height_numberbox = gr.Number(label='height', value=512, interactive=True)
                gr.Markdown(value='### Loras')
                loras_dropdown = gr.Dropdown(choices=[], label='Loras', allow_custom_value=True, interactive=True)
                loras_name_textbox = gr.Textbox(label='Name (File name without extension)')
                loras_trigger_words_dropdown = gr.Dropdown(choices=[], value=[], label='Trigger words', multiselect=True, allow_custom_value=True, interactive=True)
                loras_weight_numberbox = gr.Number(label='Weight', value=1.0, interactive=True)
                loras_caption_textbox = gr.Textbox(label='Caption (Description)')
                gr.Markdown(value='## File')
                file_path_textbox = gr.Textbox(label='Path')
                with gr.Row():
                    file_save_btn = gr.Button(value='Save')
                    file_load_btn = gr.Button(value='Load')

    return block_interface

if __name__ == '__main__':
    block_interface = main_ui()
    block_interface.queue()
    block_interface.launch(server_port=50081)