"""
scrapers/base.py: Abstract Base Class, view parser, English language detector, podcast filter,
and strict theme-page / celebrity-montage rejector.
"""

import re
from abc import ABC, abstractmethod
from typing import Generator, Dict, Any, Optional
import langdetect

# Regex patterns identifying podcasts, interviews, and multi-speaker shows
PODCAST_INTERVIEW_PATTERN = re.compile(
    r"\b(podcast\w*|pod\b|interview\w*|ep\s*\d+|episode|talkshow|radio show|guest|co-host|feat\.|ft\.)\b",
    re.IGNORECASE
)

# Common Hinglish / Romanized Hindi indicator words
HINGLISH_PATTERN = re.compile(
    r"\b(banao|jao|aao|karo|kare|karna|karne|pe|ko|se|ki|ka|ke|mein|nahi|hota|hoti|hote|hai|hain|kya|aur|toh|bhi|wala|wali|kaise|bhai|yaar|aap|tum|kar|hindi|shayari)\b",
    re.IGNORECASE
)

# Common non-Latin script ranges (Arabic, Cyrillic, Devanagari, CJK, etc.)
NON_LATIN_PATTERN = re.compile(r"[\u0400-\u04FF\u0600-\u06FF\u0900-\u097F\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]")

# Common foreign Romance language stop words (Spanish, French, Portuguese, Italian)
FOREIGN_STOPWORDS = re.compile(
    r"\b(una|uno|unos|unas|tarde|por|para|con|sin|sobre|este|esta|estos|estas|muy|bien|pero|como|donde|cuando|nada|mundo|cambiar|vida|siempre|mujer|mujeres|forma|gente|todos|todas|amigos|casa|tiempo|noche|nuevo|nueva|gracias|bueno|buena|hacer|estar|tener|decir|saber|querer|llegar|pasar|deber|creer|hablar|llevar|dejar|seguir|encontrar|llamar|venir|pensar|salir|volver|tomar|conocer|vivir|sentir|mirar|contar|esperar|buscar|entrar|trabajar|escribir|pour|dans|avec|sur|sans|mais|donc|cette|faire|avoir|etre|aller|vouloir|aimer)\b",
    re.IGNORECASE
)

# Theme page, aggregator, and faceless clip channel handle indicators
THEME_PAGE_HANDLE_PATTERN = re.compile(
    r"(aspire|billionaire|motivation|motivate|stoic|quotes|edits|clip|clips|cinema|cine|wealth|formula|mentality|hub|vibes|vault|zone|empire|capital|drops|creations|file|moment|times|society|club|facts|page|media|archive|daily|curator|central|network|official|baking|cooking|recipes|recipe|food|cake|shope|shop|store|bakery|pastry|dessert|diy|hack|tv|show|scene|relatable|memes|funny|entertainment|shorts|reelz|reels|driven|hq\b|thinkwith|mindset|luxury|attitude|sigma|rule|rules|hustle)",
    re.IGNORECASE
)

# Third-person celebrity biography / documentary narration indicators (e.g. "Sofía Vergara was on track to...")
THIRD_PERSON_BIOGRAPHY_PATTERN = re.compile(
    r"^\s*([A-Z][a-z]+ [A-Z][a-z]+ (was|is|born|turned|received|built|started|founded)|\b(In \d{4}|\(born\s+[A-Za-z]+\s+\d+|Shark Tank|Disney|Pixar|movie\b|film\b|clip\b|interview\b|Sofía Vergara|Robert Downey|Seneca|Marcus Aurelius|Kanye West|Elon Musk|Steve Jobs|Gautam Adani|Puneet Superstar|Shi Heng Yi|Balendra Shah|Anton Chekhov|Louis Pasteur|Father Lazarus|Neil deGrasse|Khabib|Banana Bread|iphone 18|Road Signs)\b)",
    re.IGNORECASE
)

# Institutional, political, military, and news organization indicators
INSTITUTIONAL_PATTERN = re.compile(
    r"(sen|rep|fbi|cia|dept|gov|police|party|news|post|times|herald|tribune|journal|review|press|army|navy|marines|airforce|guard|foundation|association|hospital|clinic|university|college|school|bank|fund|capital|corp|inc|llc|gop|dnc|democrats|republicans|fox|cnn|nbc|msnbc|bbc|npr|nyt|wsj|bloomberg|un\b|who\b)",
    re.IGNORECASE
)

def is_theme_page_or_montage(handle: str, title: str) -> bool:
    """
    Reject theme pages, faceless clip channels, institutional/news pages, and third-person celebrity montages.
    True talking-head videos feature the real creator speaking directly to the camera.
    """
    clean_handle = handle.lower().lstrip("@")
    if THEME_PAGE_HANDLE_PATTERN.search(clean_handle):
        return True
    if INSTITUTIONAL_PATTERN.search(clean_handle):
        return True
    if THIRD_PERSON_BIOGRAPHY_PATTERN.search(title):
        return True
    return False

def is_english_speaking(text: str) -> bool:
    """
    Check if a caption or title is in English.
    Returns False for foreign languages (Spanish, Indonesian, Arabic, Russian, Hindi, Hinglish, etc.).
    """
    if not text or len(text.strip()) < 4:
        return True

    if NON_LATIN_PATTERN.search(text):
        return False

    if HINGLISH_PATTERN.search(text):
        return False

    if FOREIGN_STOPWORDS.search(text):
        return False

    clean_text = re.sub(r"#\S+", "", text)
    clean_text = re.sub(r"https?://\S+", "", clean_text)
    clean_text = re.sub(r"[^\w\s]", " ", clean_text).strip()

    if len(clean_text) < 10:
        return True

    try:
        detected = langdetect.detect(clean_text)
        return detected == "en"
    except Exception:
        words = set(clean_text.lower().split())
        common_en = {"the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"}
        return len(words.intersection(common_en)) >= 1

def is_podcast_or_interview(text: str) -> bool:
    """Check if title or caption indicates a podcast, interview, or multi-person show."""
    if not text:
        return False
    return bool(PODCAST_INTERVIEW_PATTERN.search(text))

def parse_views_string(text: Optional[str]) -> int:
    """Parse textual view/like count into an integer."""
    if not text:
        return 0

    clean = text.lower().replace("views", "").replace("likes", "").replace(",", "").strip()
    match = re.search(r"([\d\.]+)\s*([kmb])?", clean)
    if not match:
        return 0

    val_str, unit = match.groups()
    try:
        val = float(val_str)
        if unit == "k":
            return int(val * 1_000)
        elif unit == "m":
            return int(val * 1_000_000)
        elif unit == "b":
            return int(val * 1_000_000_000)
        return int(val)
    except (ValueError, TypeError):
        return 0

class BaseScraper(ABC):
    def __init__(self, platform_name: str):
        self.platform_name = platform_name

    @abstractmethod
    def search_candidates(self, query: str, limit: int = 30) -> Generator[Dict[str, Any], None, None]:
        pass
