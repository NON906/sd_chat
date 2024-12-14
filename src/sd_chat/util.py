import os

def get_path_settings_file(file_name: str, new_file=False, sd_webui_extensions=False, sd_chat_dir=True):
    ret = os.path.join(os.path.dirname(__file__), '..', '..', 'settings', file_name)
    if os.path.isfile(ret):
        return ret
    ret = os.path.join('sd_chat', 'settings', file_name)
    if os.path.isfile(ret):
        return ret
    ret = os.path.join('settings', file_name)
    if os.path.isfile(ret):
        return ret
    if sd_webui_extensions:
        from modules.scripts import basedir
        from modules.paths_internal import extensions_dir
        ret = os.path.join(basedir(), 'settings', file_name)
        if os.path.isfile(ret):
            return ret
        ret = os.path.join(extensions_dir, 'sd_chat', 'settings', file_name)
        if os.path.isfile(ret):
            return ret
        ret = os.path.join(os.getcwd(), 'extensions', 'sd_chat', 'settings', file_name)
        if os.path.isfile(ret):
            return ret

    if new_file:
        if sd_webui_extensions:
            ret = os.path.join(os.path.dirname(__file__), '..', '..', 'settings', file_name)
        elif sd_chat_dir:
            ret = os.path.join('sd_chat', 'settings', file_name)
        else:
            ret = os.path.join('settings', file_name)
        os.makedirs(os.path.dirname(ret), exist_ok=True)
        return ret
    else:
        ret = os.path.join(os.path.dirname(__file__), '..', '..', 'settings_default', file_name)
        if os.path.isfile(ret):
            return ret
        ret = os.path.join('settings_default', file_name)
        if os.path.isfile(ret):
            return ret
        if sd_webui_extensions:
            from modules.scripts import basedir
            from modules.paths_internal import extensions_dir
            ret = os.path.join(basedir(), 'settings_default', file_name)
            if os.path.isfile(ret):
                return ret
            ret = os.path.join(extensions_dir, 'sd_chat', 'settings_default', file_name)
            if os.path.isfile(ret):
                return ret
            ret = os.path.join(os.getcwd(), 'extensions', 'sd_chat', 'settings_default', file_name)
            if os.path.isfile(ret):
                return ret

    return None