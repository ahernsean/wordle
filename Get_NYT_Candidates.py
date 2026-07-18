#!/usr/bin/env python3.13
import html
import json
import re
import ssl
import urllib.request

WORDLE_PAGE_URL = "https://www.nytimes.com/games/wordle/index.html"
CHUNK_URL_PATTERN = re.compile(
    r'(?:src|href)="(https://www\.nytimes\.com/games-assets/v2/(?!vendor/|metadata/)[^"]*\.js[^"]*)"'
)
MINIMUM_DICTIONARY_SIZE = 5000
WORD_ARRAY_PATTERN = re.compile(
    r'\[\s*"[a-z]{5}"\s*(?:,\s*"[a-z]{5}"\s*){' + str(MINIMUM_DICTIONARY_SIZE) + r',}\]'
)


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, context=ssl.create_default_context()) as resp:
        return resp.read().decode("utf-8", errors="replace")


def get_NYT_candidate_words() -> list[str]:
    """Scrape the NYT Wordle web client's bundled candidate dictionary.

    The dictionary is shipped to the client (unlike the answer list, which
    would spoil the game) as a JSON array literal inside one of the game's
    webpack chunks. Chunk filenames and numbering change across NYT deploys,
    so every chunk referenced from the game page is fetched and searched by
    shape (a long run of quoted 5-letter words) rather than by a fixed name.
    """
    index_html = _fetch(WORDLE_PAGE_URL)
    chunk_urls = sorted({html.unescape(u) for u in CHUNK_URL_PATTERN.findall(index_html)})

    words: list[str] = []
    for url in chunk_urls:
        chunk = _fetch(url)
        match = WORD_ARRAY_PATTERN.search(chunk)
        if match and len(match.group(0)) > len("".join(words)):
            words = json.loads(match.group(0))

    return sorted(set(words))


if __name__ == "__main__":
    words = get_NYT_candidate_words()
    print(f"Found {len(words)} words")
    with open("wordle.txt", "w") as f:
        f.write("\n".join(words))
