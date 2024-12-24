import json
import os
import uuid
import urllib.parse
import glob
import hashlib
import aiohttp
import asyncio

from .util import get_path_settings_file

async def civitai_fetch(url):
    with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
        settings_dict = json.load(f)
    headers = {}
    if 'civitai_api_key' in settings_dict:
        headers = {"Authorization": f"Bearer {settings_dict['civitai_api_key']}"}
    else:
        civitai_api_key = os.environ.get('CIVITAI_API_KEY')
        if civitai_api_key is not None:
            headers = {"Authorization": f"Bearer {civitai_api_key}"}
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
                    'base_model': version['baseModel'],
                    'description': version['description'],
                }
            else:
                ret_dict = {
                    'version_id': version['id'],
                    'base_model': version['baseModel'],
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
        ret['version_base_model'] = self.version_dict[str(version_id)]['baseModel']

        return ret

    def __rewrite_settings(self, version_id: int, base_model_name: str, file_name: str, caption: str, weight: float):
        with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
            settings_dict = json.load(f)

        if self.version_dict[str(version_id)]['model']['type'] == 'Checkpoint':
            name = self.version_dict[str(version_id)]['model']['name']
            loop = 1
            while name in settings_dict['checkpoints']:
                name = f'{self.version_dict[str(version_id)]['model']['name']} ({loop})'
                loop += 1
            settings_dict['checkpoints'][name] = {}
            settings_dict['checkpoints'][name]['name'] = os.path.splitext(file_name)[0]
            settings_dict['checkpoints'][name]['caption'] = caption
            settings_dict['checkpoints'][name]['base_model'] = self.version_dict[str(version_id)]['base_model']
        elif self.version_dict[str(version_id)]['model']['type'] == 'LORA':
            name = self.version_dict[str(version_id)]['model']['name']
            loop = 1
            while name in settings_dict['checkpoints'][base_model_name]['loras']:
                name = f'{self.version_dict[str(version_id)]['model']['name']} ({loop})'
                loop += 1
            settings_dict['checkpoints'][base_model_name]['loras'][name] = {}
            settings_dict['checkpoints'][base_model_name]['loras'][name]['name'] = os.path.splitext(file_name)[0]
            settings_dict['checkpoints'][base_model_name]['loras'][name]['trigger_words'] = self.version_dict[str(version_id)]['trainedWords']
            settings_dict['checkpoints'][base_model_name]['loras'][name]['weight'] = weight
            settings_dict['checkpoints'][base_model_name]['loras'][name]['caption'] = caption
            settings_dict['checkpoints'][base_model_name]['loras'][name]['base_model'] = self.version_dict[str(version_id)]['baseModel']

        with open(get_path_settings_file('settings.json', new_file=True), 'w', encoding="utf-8") as f:
            json.dump(settings_dict, f, indent=2)

    async def install_model(self, checkpoints_path: str | None, loras_path: str | None, version_id: int | None, caption: str, base_model_name: str = None, weight: float = 1.0):
        with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
            settings_dict = json.load(f)

        if 'apis' in settings_dict and 'webui_client' in settings_dict['apis'] and 'host' in settings_dict['apis']['webui_client']:
            if settings_dict['apis']['webui_client']['host'] != 'localhost' and settings_dict['apis']['webui_client']['host'] != '127.0.0.1':
                return None

        if version_id is not None:
            if not str(version_id) in self.version_dict:
                self.version_dict[str(version_id)] = await civitai_fetch(f'https://civitai.com/api/v1/model-versions/{version_id}')

            if 'model' in self.version_dict[str(version_id)] and 'type' in self.version_dict[str(version_id)]['model'] and self.version_dict[str(version_id)]['model']['type'] == 'LORA' and not base_model_name in settings_dict['checkpoints']:
                return None

        if version_id is None or 'model' in self.version_dict[str(version_id)] and 'type' in self.version_dict[str(version_id)]['model'] and self.version_dict[str(version_id)]['model']['type'] == 'LORA' and 'not_installed' in settings_dict['checkpoints'][base_model_name]:
            if 'checkpoints_path' in settings_dict:
                download_checkpoints_path = settings_dict['checkpoints_path']
            elif checkpoints_path is not None:
                download_checkpoints_path = checkpoints_path
            else:
                download_checkpoints_path = os.path.join(settings_dict['save_path'], 'models', 'StableDiffusion')
            if os.path.isfile(os.path.join(download_checkpoints_path, settings_dict['checkpoints'][base_model_name]['name'] + '.safetensors')):
                if 'not_installed' in settings_dict['checkpoints'][base_model_name]:
                    del settings_dict['checkpoints'][base_model_name]['not_installed']
                    with open(get_path_settings_file('settings.json', new_file=True), 'w', encoding="utf-8") as f:
                        json.dump(settings_dict, f, indent=2)

        download_id = str(uuid.uuid4())

        async def download_task_main(version_id: int):
            with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
                settings_dict = json.load(f)

            for file_dict in self.version_dict[str(version_id)]['files']:
                if file_dict['downloadUrl'] == self.version_dict[str(version_id)]['downloadUrl']:
                    file_name_default = file_dict['name']
                    hash_sha256 = file_dict['hashes']['SHA256']

            if self.version_dict[str(version_id)]['model']['type'] == 'Checkpoint':
                check_path = checkpoints_path
            elif self.version_dict[str(version_id)]['model']['type'] == 'LORA':
                check_path = loras_path
            already_downloaded_list = glob.glob(os.path.join(check_path, '**', '*'), recursive=True)
            for already_file in already_downloaded_list:
                if os.path.isfile(already_file):
                    with open(already_file, 'rb') as f:
                        if str(hashlib.sha256(f.read()).hexdigest()).lower() == hash_sha256.lower():
                            return already_file

            headers = {}
            if 'civitai_api_key' in settings_dict:
                headers = {"Authorization": f"Bearer {settings_dict['civitai_api_key']}"}
            else:
                civitai_api_key = os.environ.get('CIVITAI_API_KEY')
                if civitai_api_key is not None:
                    headers = {"Authorization": f"Bearer {civitai_api_key}"}
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(self.version_dict[str(version_id)]['downloadUrl']) as response:
                    download_data = await response.read()
            
            if self.version_dict[str(version_id)]['model']['type'] == 'Checkpoint':
                if 'checkpoints_path' in settings_dict:
                    download_write_path = settings_dict['checkpoints_path']
                elif checkpoints_path is not None:
                    download_write_path = checkpoints_path
                else:
                    download_write_path = os.path.join(settings_dict['save_path'], 'models', 'StableDiffusion')
            elif self.version_dict[str(version_id)]['model']['type'] == 'LORA':
                if 'lora_path' in settings_dict:
                    download_write_path = settings_dict['lora_path']
                elif loras_path is not None:
                    download_write_path = loras_path
                else:
                    download_write_path = os.path.join(settings_dict['save_path'], 'models', 'Lora')
            os.makedirs(download_write_path, exist_ok=True)
            
            file_name = file_name_default
            file_full_path = os.path.join(download_write_path, file_name)
            if os.path.isfile(file_full_path):
                file_index = 0
                while os.path.isfile(file_full_path):
                    base_file_name, ext = os.path.splitext(file_name_default)
                    file_name = f'{base_file_name}_{file_index}{ext}'
                    file_full_path = os.path.join(download_write_path, file_name)
                    file_index += 1

            with open(file_full_path, 'wb') as f:
                f.write(download_data)

            return file_full_path

        async def download_task():
            with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
                settings_dict = json.load(f)

            self.download_ids.append(download_id)

            if version_id is None or 'model' in self.version_dict[str(version_id)] and 'type' in self.version_dict[str(version_id)]['model'] and self.version_dict[str(version_id)]['model']['type'] == 'LORA' and 'not_installed' in settings_dict['checkpoints'][base_model_name]:
                version_id_base = settings_dict['checkpoints'][base_model_name]['version_id']
                if not str(version_id_base) in self.version_dict:
                    self.version_dict[str(version_id_base)] = await civitai_fetch(f'https://civitai.com/api/v1/model-versions/{version_id_base}')
                file_full_path = await download_task_main(version_id_base)
                with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
                    settings_dict = json.load(f)
                processed_file_name = os.path.splitext(os.path.basename(file_full_path))[0]
                if processed_file_name != settings_dict['checkpoints'][base_model_name]['name']:
                    settings_dict['checkpoints'][base_model_name]['name'] = processed_file_name
                if 'not_installed' in settings_dict['checkpoints'][base_model_name]:
                    del settings_dict['checkpoints'][base_model_name]['not_installed']
                with open(get_path_settings_file('settings.json', new_file=True), 'w', encoding="utf-8") as f:
                    json.dump(settings_dict, f, indent=2)

            if version_id is not None:
                file_full_path = await download_task_main(version_id)
                self.__rewrite_settings(version_id, base_model_name, os.path.basename(file_full_path), caption, weight)

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

    async def search(self, query: str, page: int = 0):
        search_results = await civitai_fetch(f'https://civitai.com/api/v1/models?limit=10&page={(page + 1)}&query={urllib.parse.quote(query)}&types=Checkpoint&types=LORA')
        ret = []
        for result in search_results['items']:
            ret_item = {
                'model_id': result['id'],
                'name': result['name'],
                #'description': result['description'],
                'type': result['type'],
            }
            ret.append(ret_item)
            self.model_dict[str(result['id'])] = result
            for version in self.model_dict[str(result['id'])]['modelVersions']:
                self.version_dict[str(version['id'])] = version
        return ret
