""" Bot factory"""
from common.settings import Settings
from common.utils import Folder, sub_file
from .bot import Bot, GameMode
from .mahjongtrio import BotMahjongTrioAI


MODEL_TYPE_STRINGS = ["MahjongTrioAI", "Local", "AkagiOT", "MJAPI"]


def get_bot(settings:Settings) -> Bot:
    """ create the Bot instance based on settings"""
    
    match settings.model_type:
        case "MahjongTrioAI":
            bot = BotMahjongTrioAI(
                settings.trio_model_file,
                settings.trio_device,
                settings.trio_action_mode,
            )
        case "Local":   
            from .local.bot_local import BotMortalLocal
            model_files:dict = {
                GameMode.MJ4P: sub_file(Folder.MODEL, settings.model_file),
                GameMode.MJ3P: sub_file(Folder.MODEL, settings.model_file_3p)
            }
            bot = BotMortalLocal(model_files)
        case "AkagiOT":
            from .akagiot.bot_akagiot import BotAkagiOt
            bot = BotAkagiOt(settings.akagi_ot_url, settings.akagi_ot_apikey)
        case "MJAPI":
            from .mjapi.bot_mjapi import BotMjapi
            bot = BotMjapi(settings)
        case _:
            raise ValueError(f"Unknown model type: {settings.model_type}")

    return bot

