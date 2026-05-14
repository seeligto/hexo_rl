"""Public bot modules (§176 P78d).

Canonical home for bot wrappers. Previously lived at hexo_rl.bootstrap.bots.
"""

from hexo_rl.bots.community_api_bot import CommunityAPIBot
from hexo_rl.bots.krakenbot_bot import KrakenBotBot
from hexo_rl.bots.our_model_bot import OurModelBot
from hexo_rl.bots.random_bot import RandomBot
from hexo_rl.bots.sealbot_bot import SealBotBot

__all__ = [
    "CommunityAPIBot",
    "KrakenBotBot",
    "OurModelBot",
    "RandomBot",
    "SealBotBot",
]
