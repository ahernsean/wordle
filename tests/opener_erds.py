"""Test support: store opener ERDs the way the swarm does at completion."""

from cache_sqlite import ScoreCache
from erd_reduction import reduce_opener
from wordle_engine import ERD_ALL, GAME_GUESSES, ResponseCache, load_word_list


def store_opener_erds(cache_path, answers, candidate_list_path):
    """Reduce each candidate and store the ERD of every one that completes.

    A report reads these rows and never writes them, so a test that wants a
    ranking puts them there the way the worker that finished each opener
    would.  Returns the openers stored.
    """
    cache = ScoreCache(cache_path, answers, checkpoint_on_close=False)
    stored = []
    try:
        response_cache = ResponseCache(answers, score_cache=cache)
        for word in load_word_list(candidate_list_path):
            reduction = reduce_opener(
                word, answers, response_cache, cache, ERD_ALL, GAME_GUESSES - 1)
            if reduction["state"] == "complete":
                cache.write_opener_erd(
                    word, ERD_ALL, reduction["erd"],
                    reduction["max_remaining_depth"],
                    reduction["response_group_count"])
                stored.append(word)
    finally:
        cache.close()
    return stored
