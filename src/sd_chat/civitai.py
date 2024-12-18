import json
import os
import uuid
import aiohttp
import asyncio

from .util import get_path_settings_file

async def civitai_fetch(url):
    with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
        settings_dict = json.load(f)
    headers = {}
    if 'civitai_api_key' in settings_dict:
        headers = {"Authorization": f"Bearer {settings_dict['civitai_api_key']}"}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as response:
            return await response.json()

class CivitaiAPI():
    model_dict = {}
    version_dict = {}
    download_ids = []
    finished_ids = []

    async def get_model_versions(self, model_id: int) -> list:
        if not str(model_id) in self.model_dict:
           self.model_dict[str(model_id)] = await civitai_fetch(f'https://civitai.com/api/v1/models/{model_id}')

        ret_list = []
        for version in self.model_dict[str(model_id)]['modelVersions']:
            self.version_dict[str(version['id'])] = version
            if 'description' in version:
                ret_dict = {
                    'version_id': version['id'],
                    'name': version['name'],
                    'description': version['description'],
                }
            else:
                ret_dict = {
                    'version_id': version['id'],
                    'name': version['name'],
                }
            ret_list.append(ret_dict)
        return ret_list

    async def get_model_info(self, version_id: int):
        if not str(version_id) in self.version_dict or not 'modelId' in self.version_dict[str(version_id)] or not self.version_dict[str(version_id)] in self.model_dict:
            self.version_dict[str(version_id)] = await civitai_fetch(f'https://civitai.com/api/v1/model-versions/{version_id}')
            self.model_dict[str(self.version_dict[str(version_id)]['modelId'])] = await civitai_fetch(f'https://civitai.com/api/v1/models/{self.version_dict[str(version_id)]['modelId']}')

        model_id = str(self.version_dict[str(version_id)]['modelId'])
        ret = {}
        ret['model_name'] = self.model_dict[model_id]['name']
        ret['model_description'] = self.model_dict[model_id]['description']
        ret['version_name'] = self.version_dict[str(version_id)]['name']
        if 'description' in self.version_dict[str(version_id)]:
            ret['version_description'] = self.version_dict[str(version_id)]['description']
        ret['type'] = self.model_dict[model_id]['type']

        return ret

    async def install_model(self, version_id: int, caption: str, base_model_name: str = None, weight: float = 1.0):
        if not str(version_id) in self.version_dict:
            self.version_dict[str(version_id)] = await civitai_fetch(f'https://civitai.com/api/v1/model-versions/{version_id}')

        with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
            settings_dict = json.load(f)
        if self.version_dict[str(version_id)]['model']['type'] == 'LORA' and not base_model_name in settings_dict['checkpoints']:
            return None

        download_id = str(uuid.uuid4())
        async def download_task():
            self.download_ids.append(download_id)
                
            headers = {}
            if 'civitai_api_key' in settings_dict:
                headers = {"Authorization": f"Bearer {settings_dict['civitai_api_key']}"}
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(self.version_dict[str(version_id)]['downloadUrl']) as response:
                    download_data = await response.read()
            if 'models_path' in settings_dict:
                if self.version_dict[str(version_id)]['model']['type'] == 'Checkpoint':
                    download_write_path = os.path.join(settings_dict['models_path'], 'StableDiffusion')
                elif self.version_dict[str(version_id)]['model']['type'] == 'LORA':
                    download_write_path = os.path.join(settings_dict['models_path'], 'Lora')
            else:
                if self.version_dict[str(version_id)]['model']['type'] == 'Checkpoint':
                    download_write_path = os.path.join(settings_dict['save_path'], 'models', 'StableDiffusion')
                elif self.version_dict[str(version_id)]['model']['type'] == 'LORA':
                    download_write_path = os.path.join(settings_dict['save_path'], 'models', 'Lora')
            os.makedirs(download_write_path, exist_ok=True)
            file_name = self.version_dict[str(version_id)]['files'][0]['name']
            with open(os.path.join(download_write_path, file_name), 'wb') as f:
                f.write(download_data)

            with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
                settings_dict = json.load(f)

            if self.version_dict[str(version_id)]['model']['type'] == 'Checkpoint':
                name = self.version_dict[str(version_id)]['model']['name']
                if not name in settings_dict['checkpoints']:
                    settings_dict['checkpoints'][name] = {}
                settings_dict['checkpoints'][name]['name'] = os.path.splitext(file_name)[0]
                settings_dict['checkpoints'][name]['caption'] = caption
            elif self.version_dict[str(version_id)]['model']['type'] == 'LORA':
                name = self.version_dict[str(version_id)]['model']['name']
                if not name in settings_dict['checkpoints'][base_model_name]['loras']:
                    settings_dict['checkpoints'][base_model_name]['loras'] = {}
                settings_dict['checkpoints'][base_model_name]['loras'][name] = {}
                settings_dict['checkpoints'][base_model_name]['loras'][name]['name'] = os.path.splitext(file_name)[0]
                settings_dict['checkpoints'][base_model_name]['loras'][name]['trigger_words'] = self.version_dict[str(version_id)]['trainedWords']
                settings_dict['checkpoints'][base_model_name]['loras'][name]['weight'] = weight
                settings_dict['checkpoints'][base_model_name]['loras'][name]['caption'] = caption

            with open(get_path_settings_file('settings.json', new_file=True), 'w', encoding="utf-8") as f:
                json.dump(settings_dict, f, indent=2)

            self.download_ids.remove(download_id)
            self.finished_ids.append(download_id)

        asyncio.create_task(download_task())

        return download_id

    def download_status(self, download_id: str):
        if download_id in self.download_ids:
            return 'Downloading.'
        elif download_id in self.finished_ids:
            return 'Finished.'
        else:
            return 'Nothing.'