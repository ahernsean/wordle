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
    Raises if no chunk contains a qualifying array, rather than returning an
    empty result a caller might mistake for "no candidates".
    """
    index_html = _fetch(WORDLE_PAGE_URL)
    chunk_urls = sorted({html.unescape(u) for u in CHUNK_URL_PATTERN.findall(index_html)})

    words: list[str] = []
    for url in chunk_urls:
        chunk = _fetch(url)
        for match in WORD_ARRAY_PATTERN.finditer(chunk):
            candidate = json.loads(match.group(0))
            if len(candidate) > len(words):
                words = candidate

    if not words:
        raise RuntimeError(
            f"No candidate dictionary (a run of >= {MINIMUM_DICTIONARY_SIZE} "
            f"quoted 5-letter words) found in any chunk referenced from "
            f"{WORDLE_PAGE_URL}. The page's chunk structure may have changed."
        )

    return sorted(set(words))


def _validate_against_answer_list(candidate_words: list[str]) -> None:
    """Plausibility gate: every known answer word must be a legal guess.

    A partial or corrupted extraction is far more likely to silently drop
    words than to invent new well-formed ones, so a missing answer word is
    the cheapest available signal that the scrape should not be trusted.
    """
    try:
        with open("NYT_wordlist.txt") as f:
            answer_words = f.read().split()
    except FileNotFoundError:
        return
    missing = sorted(set(answer_words) - set(candidate_words))
    if missing:
        raise RuntimeError(
            f"{len(missing)} known answer word(s) are missing from the "
            f"scraped candidate dictionary (e.g. {missing[:5]}) — treating "
            "this as a corrupted or partial extraction rather than writing it."
        )


if __name__ == "__main__":
    words = get_NYT_candidate_words()
    _validate_against_answer_list(words)
    print(f"Found {len(words)} words")
    with open("wordle.txt", "w") as f:
        f.write("\n".join(words))
