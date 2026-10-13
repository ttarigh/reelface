"""
agent.py: Autonomous Orchestrator Agent for Discovering Top 500 Viral Talking-Head Instagram Reels.
Focuses strictly on REAL HUMAN TALKING-HEAD CREATORS:
- Prioritizes personal creator profiles (solitary direct-to-camera speakers)
- Self-expanding graph discovery (extracts suggested/similar creators from each profile)
- Full browser control via Playwright (with sleep prevention via caffeinate)
- Deep Learning Face & Landmark Detection with OpenCV YuNet
- Rejects faceless theme pages, celebrity photo montages, podcasts, and foreign language clips
- Enforces 500 unique creators (strictly 1 highest-engagement reel per creator)
- Strict Virality Gate: >= 100K Likes (1M+ views)
"""

import os
import sys
import time
import argparse
import signal
from collections import deque
from typing import List, Dict, Any, Set

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel

from browser_engine import BrowserEngine
from face_verifier import FaceVerifier
from storage import VideoDatabase, format_number
from scrapers.instagram_scraper import InstagramScraper, IG_POPULAR_TOPICS, IG_CURATED_CREATORS
from scrapers.base import is_theme_page_or_montage

console = Console()

class ViralFaceAgent:
    def __init__(
        self,
        target_creators: int = 500,
        headless: bool = False,
        min_confidence: float = 0.85,
        min_likes: int = 100_000,
        output_md: str = "top_500_viral_talking_heads.md"
    ):
        self.target_creators = target_creators
        self.headless = headless
        self.min_confidence = min_confidence
        self.min_likes = min_likes
        self.output_md = output_md

        self.db = VideoDatabase()
        self.face_verifier = FaceVerifier(min_confidence=self.min_confidence)
        self.browser_engine: BrowserEngine | None = None
        self.is_running = True

        # Statistics
        self.total_candidates_examined = 0
        self.total_faces_verified = 0
        self.duplicates_skipped = 0
        self.below_virality_skipped = 0

    def start_browser(self):
        """Initialize persistent Playwright browser."""
        console.print("[cyan]Launching Playwright Browser Engine (Stealth mode)...[/cyan]")
        self.browser_engine = BrowserEngine(headless=self.headless)
        self.browser_engine.start()

    def shutdown(self):
        """Clean shutdown handler."""
        self.is_running = False
        if self.browser_engine:
            self.browser_engine.close()
        md_path = self.db.export_markdown(self.output_md, limit=self.target_creators)
        stats = self.db.get_stats()
        console.print(f"\n[bold green]Saved current progress ({stats['total_creators']} unique creators) to {md_path}[/bold green]")

    def run(self):
        """Main autonomous execution loop."""
        initial_count = self.db.get_count()
        console.print(Panel.fit(
            f"[bold magenta]🎬 Genuine Talking-Head Instagram Reels Agent[/bold magenta]\n"
            f"[bold]Target Goal:[/bold] [yellow]{self.target_creators} Unique Talking-Head Creators[/yellow]\n"
            f"[bold]Current DB Count:[/bold] [green]{initial_count} Clean Creators[/green]\n"
            f"[bold]Virality Floor:[/bold] [bold red]>={format_number(self.min_likes)} Likes[/bold red] (~{format_number(self.min_likes * 10)} Views)\n"
            f"[bold]Browser Mode:[/bold] {'Headless' if self.headless else 'Interactive Headful'}\n"
            f"[bold]Strict Filter:[/bold] Real Human Speakers Only (Theme pages & celebrity montages banned)\n"
            f"[bold]Computer Vision:[/bold] OpenCV YuNet Face & Landmark Detection (Confidence >= {self.min_confidence})",
            title="System Status"
        ))

        if initial_count >= self.target_creators:
            console.print(f"[bold green]Target already met ({initial_count}/{self.target_creators})![/bold green]")
            self.db.export_markdown(self.output_md, limit=self.target_creators)
            return

        self.start_browser()
        scraper = InstagramScraper(self.browser_engine, min_likes=self.min_likes)

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[green]{task.completed}/{task.total} creators"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Collecting Unique Creators", total=self.target_creators, completed=initial_count)

            # Stage 1: Creator Graph Traversal (The primary engine for 100% genuine talking heads)
            creator_queue = deque(IG_CURATED_CREATORS)
            seen_handles: Set[str] = set([h.lower() for h in IG_CURATED_CREATORS])

            while creator_queue and self.db.get_count() < self.target_creators and self.is_running:
                creator = creator_queue.popleft()
                clean_handle = creator.strip().lstrip('@').lower()
                if is_theme_page_or_montage(clean_handle, ""):
                    continue

                # Skip immediately if we already have this creator verified!
                if self.db.has_creator(clean_handle):
                    continue

                progress.update(task, description=f"[bold]Scanning Creator Profile:[/] @{clean_handle}...")
                found_new = self._process_candidates(scraper.scrape_creator_reels(clean_handle, limit=8), progress, task)

                # Discover similar creators from this profile to expand the talking-head graph
                if len(creator_queue) < 200:
                    similar = scraper.discover_similar_creators(clean_handle)
                    for sim_handle in similar:
                        if sim_handle not in seen_handles and not is_theme_page_or_montage(sim_handle, ""):
                            seen_handles.add(sim_handle)
                            creator_queue.append(sim_handle)

            # Stage 2: Pure English Talking-Head Topics (storytime, monologue, advice)
            if self.db.get_count() < self.target_creators and self.is_running:
                for topic in IG_POPULAR_TOPICS:
                    if self.db.get_count() >= self.target_creators or not self.is_running:
                        break
                    progress.update(task, description=f"[bold]Scraping Monologue Grid:[/] #{topic}...")
                    self._process_candidates(scraper.search_candidates(topic, limit=25), progress, task)

        # Final wrap-up
        final_count = self.db.get_count()
        md_file = self.db.export_markdown(self.output_md, limit=self.target_creators)
        stats = self.db.get_stats()

        console.print(Panel.fit(
            f"[bold green]✨ Scraping & Verification Complete![/bold green]\n"
            f"[bold]Total Unique Creators Verified:[/bold] [yellow]{final_count} / {self.target_creators}[/yellow]\n"
            f"[bold]Total Aggregate Likes:[/bold] [green]{format_number(stats['total_likes'])}[/green]\n"
            f"[bold]Total Estimated Views:[/bold] [green]{format_number(stats['total_views'])}[/green]\n"
            f"[bold]Markdown Export:[/bold] [cyan]{md_file}[/cyan]\n"
            f"[bold]Candidates evaluated:[/bold] {self.total_candidates_examined} | "
            f"[bold]Verified faces:[/bold] {self.total_faces_verified} | "
            f"[bold]Below 100K Likes rejected:[/bold] {self.below_virality_skipped}",
            title="Execution Report"
        ))

    def _process_candidates(self, candidates, progress, task) -> bool:
        """Process, verify with CV, deduplicate, and record candidates."""
        added = False
        for cand in candidates:
            if not self.is_running or self.db.get_count() >= self.target_creators:
                break

            self.total_candidates_examined += 1

            # Strict Virality Floor
            likes = cand.get("like_count", 0)
            if likes < self.min_likes:
                self.below_virality_skipped += 1
                continue

            thumb_url = cand.get("thumbnail_url")
            if not thumb_url:
                continue

            # Strict CV Face & Landmark Detection
            is_talking_head, metrics = self.face_verifier.verify_image_url(thumb_url)
            if not is_talking_head:
                continue

            cand["face_confidence"] = metrics.get("confidence", 0.0)
            cand["face_area_pct"] = metrics.get("face_area_ratio_pct", 0.0)
            cand["verified_talking_head"] = True

            # Upsert into database
            status = self.db.upsert_video(cand)
            current_count = self.db.get_count()

            if status == "NEW":
                added = True
                progress.update(task, completed=min(current_count, self.target_creators))
                if current_count % 3 == 0 or current_count == self.target_creators:
                    self.db.export_markdown(self.output_md, limit=self.target_creators)

                console.print(
                    f"  [green]✓ NEW[/green] [bold]{cand['creator_name'][:25]}[/bold] "
                    f"| {cand['title'][:32]}... | [red]{format_number(cand['like_count'])} Likes[/red] "
                    f"([yellow]~{format_number(cand['view_count'])} Views[/yellow]) "
                    f"| Face: [cyan]{int(cand['face_confidence']*100)}%[/cyan]"
                )
                break  # Stop checking further reels for this creator once verified!
            elif status == "UPDATED":
                console.print(
                    f"  [cyan]↑ UPDATED[/cyan] [bold]{cand['creator_name'][:25]}[/bold] "
                    f"with higher engagement: [red]{format_number(cand['like_count'])} Likes[/red]"
                )
            else:
                self.duplicates_skipped += 1
        return added

def main():
    parser = argparse.ArgumentParser(description="Autonomous Scraping Agent for Genuine Instagram Talking-Head Creators")
    parser.add_argument("--target", type=int, default=500, help="Target number of unique creators (default: 500)")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode (default: False)")
    parser.add_argument("--min-confidence", type=float, default=0.85, help="Minimum CV face detection confidence (default: 0.85)")
    parser.add_argument("--min-likes", type=int, default=100000, help="Minimum Like threshold (default: 100,000 Likes ≈ 1M+ views)")
    parser.add_argument("--output", type=str, default="top_500_viral_talking_heads.md", help="Output markdown filepath")

    args = parser.parse_args()

    agent = ViralFaceAgent(
        target_creators=args.target,
        headless=args.headless,
        min_confidence=args.min_confidence,
        min_likes=args.min_likes,
        output_md=args.output
    )

    def sig_handler(sig, frame):
        console.print("\n[yellow]Received interrupt signal, saving progress and exiting cleanly...[/yellow]")
        agent.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    try:
        agent.run()
    finally:
        agent.shutdown()

if __name__ == "__main__":
    main()
