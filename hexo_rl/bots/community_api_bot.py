"""Re-export shim for CommunityAPIBot (§176 P78a).

Implementation lives in hexo_rl.bootstrap.bots.community_api_bot until P78d.
"""

from hexo_rl.bootstrap.bots.community_api_bot import CommunityAPIBot

__all__ = ["CommunityAPIBot"]
