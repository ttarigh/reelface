"""
scrapers/youtube_scraper.py: High-yield YouTube Shorts scraper for talking-head videos.
Combines:
1. Search queries sorted strictly by view count (sp=CAMSAhAB) for true viral reach.
2. Curated channels known for viral talking-head monologues, podcasts, interviews, and commentary.
"""

import os
import time
import urllib.parse
from typing import Generator, Dict, Any, List, Optional
import yt_dlp
from scrapers.base import BaseScraper
from browser_engine import BrowserEngine

# Top search queries for talking head formats across podcast, commentary, advice & monologue
YOUTUBE_TALKING_HEAD_QUERIES = [
    "podcast clips shorts",
    "interview clips shorts",
    "monologue commentary shorts",
    "the diary of a ceo shorts",
    "huberman lab shorts",
    "lex fridman shorts",
    "joe rogan podcast shorts",
    "chris williamson modern wisdom shorts",
    "shawn ryan show shorts",
    "jordan peterson advice shorts",
    "mel robbins shorts",
    "jay shetty shorts",
    "dr k healthy gamer shorts",
    "tom bilyeu impact theory shorts",
    "cleo abram shorts",
    "ali abdaal productivity shorts",
    "alex hormozi business advice shorts",
    "graham stephan finance shorts",
    "dan koe shorts",
    "codie sanchez business shorts",
    "noah kagan shorts",
    "mark manson advice shorts",
    "ryan holiday daily stoic shorts",
    "neil degrasse tyson star talk shorts",
    "brian cox physics shorts",
    "legal eagle law breakdown shorts",
    "doctor mike medical facts shorts",
    "doctor eric berg health shorts",
    "simon sinek leadership shorts",
    "gary vee motivation shorts",
    "psychology facts monologue shorts",
    "life advice monologue shorts",
    "dating advice coach shorts",
    "career advice monologue shorts",
    "philosophy stoicism shorts",
    "history facts commentary shorts",
    "movie review film essay shorts",
    "social skills conversation tips shorts",
    "public speaking confidence shorts",
    "book summary lessons shorts",
    "fitness coach talking head shorts",
    "nutrition facts doctor shorts",
    "deep thoughts storytelling shorts",
    "tech review monologue shorts",
    "investing lessons finance shorts",
    "crime story monologue shorts",
    "standup crowdwork clip shorts",
    "business breakdown talking head shorts"
]

CURATED_CHANNELS = [
    "@hubermanlab",
    "@lexclips",
    "@lexfridman",
    "@joerogan",
    "@ChrisWillx",
    "@TheDiaryOfACEO",
    "@melrobbins",
    "@JordanBPeterson",
    "@jayshetty",
    "@shawnryanshow",
    "@TomBilyeu",
    "@valuetainment",
    "@cleoabram",
    "@AliAbdaal",
    "@alexhormozi",
    "@DanKoeTalks",
    "@GrahamStephan",
    "@AndreiJikh",
    "@CodieSanchezCT",
    "@NoahKagan",
    "@DailyStoic",
    "@markmanson",
    "@NeildeGrasseTyson",
    "@LegalEagle",
    "@DoctorMike",
    "@DrEricBergDC",
    "@SimonSinek",
    "@garyvee",
    "@HealthyGamerGG",
    "@AndrewSchulz",
    "@JohnnyHarris",
    "@Veritasium",
    "@MarkRober",
    "@EconomicsExplained",
    "@ColdFusion",
    "@TED",
    "@BigThink",
    "@SchoolofLifeChannel",
    "@AcademyofIdeas",
    "@Einzelganger",
    "@Wisecrack",
    "@Nerdwriter1",
    "@FilmCourage",
    "@CallHerDaddy",
    "@TrashTaste",
    "@SomeOrdinaryGamers",
    "@MoistCr1tikal",
    "@Ludwig",
    "@MeetKevin",
    "@HumphreyYang",
    "@BrianJung",
    "@MinorityMindset",
    "@NateOBrien",
    "@ThomasFrank",
    "@MattDAvella",
    "@NathanielDrew",
    "@BetterIdeas",
    "@RajShamani",
    "@BeerBiceps",
    "@TanmayBhat",
    "@AnkurWarikoo",
    "@AkshatShrivastava"
]

class YouTubeScraper(BaseScraper):
    def __init__(self, browser_engine: Optional[BrowserEngine] = None):
        super().__init__("YouTube Shorts")
        self.engine = browser_engine
        self.ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True
        }

    def search_candidates(self, query: str, limit: int = 50) -> Generator[Dict[str, Any], None, None]:
        """
        Search YouTube for viral Shorts matching query.
        Automatically applies sort-by-view-count (sp=CAMSAhAB) if not a direct URL.
        """
        if query.startswith("http://") or query.startswith("https://"):
            target_url = query
        else:
            encoded_query = urllib.parse.quote_plus(query)
            target_url = f"https://www.youtube.com/results?search_query={encoded_query}&sp=CAMSAhAB"

        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(target_url, download=False)
                if not info or "entries" not in info:
                    return

                count = 0
                for entry in info["entries"]:
                    cand = self._parse_entry(entry)
                    if cand:
                        yield cand
                        count += 1
                        if count >= limit:
                            break

        except Exception as e:
            print(f"[YouTubeScraper Error on '{query}']: {e}")

    def scrape_channel_shorts(self, channel_handle: str, limit: int = 15) -> Generator[Dict[str, Any], None, None]:
        """Scrape the most popular shorts from a creator channel."""
        clean_handle = channel_handle if channel_handle.startswith("@") else f"@{channel_handle}"
        url = f"https://www.youtube.com/{clean_handle}/shorts"

        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info or "entries" not in info:
                    return

                count = 0
                for entry in info["entries"]:
                    cand = self._parse_entry(entry)
                    if cand:
                        if not cand.get("creator_name") or cand["creator_name"] == "YouTubeCreator":
                            cand["creator_name"] = clean_handle
                        cand["creator_url"] = f"https://www.youtube.com/{clean_handle}"
                        yield cand
                        count += 1
                        if count >= limit:
                            break
        except Exception as e:
            pass  # Some channels might have alternate URL formats or no shorts

    def _parse_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse entry and validate short duration."""
        if not entry:
            return None

        duration = entry.get("duration")
        if duration and duration > 180:
            return None

        video_id = entry.get("id") or ""
        video_url = entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
        title = entry.get("title") or "YouTube Short"
        channel_name = entry.get("channel") or entry.get("uploader") or ""
        channel_url = entry.get("uploader_url") or entry.get("channel_url") or ""
        views = int(entry.get("view_count") or 0)
        upload_date = str(entry.get("upload_date") or "2025-2026")

        thumbnail_url = ""
        thumbnails = entry.get("thumbnails", [])
        if thumbnails:
            thumbnail_url = thumbnails[-1].get("url", "")
        if not thumbnail_url and video_id:
            thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

        short_url = f"https://www.youtube.com/shorts/{video_id}" if video_id else video_url

        return {
            "platform": self.platform_name,
            "creator_name": channel_name,
            "creator_url": channel_url,
            "video_id": video_id,
            "video_url": short_url,
            "title": title,
            "view_count": views,
            "like_count": 0,
            "upload_date": upload_date,
            "thumbnail_url": thumbnail_url
        }
