import json
import aiohttp

from .util import get_path_settings_file

async def civitai_fetch(url):
    with open(get_path_settings_file('settings.json'), 'r', encoding="utf-8") as f:
        settings_dict = json.load(f)
    headers = {}
    if settings_dict['civitai_api_key'] is not None:
        headers = {"Authorization": f"Bearer {settings_dict['civitai_api_key']}"}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as response:
            return await response.json()

class CivitaiAPI():
    model_dict = {}

    async def get_model_versions(self, model_id: int) -> list:
        if not str(model_id) in self.model_dict:
           self.model_dict[str(model_id)] = await civitai_fetch(f'https://civitai.com/api/v1/models/{model_id}')

        ret_list = []
        for version in self.model_dict[str(model_id)]['modelVersions']:
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