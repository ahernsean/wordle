from bs4 import BeautifulSoup
import ssl
import urllib.request
import re

def get_NYT_answer_words() -> list[str]:
    req = urllib.request.Request(
        "https://wordletools.azurewebsites.net/weightedbottles",
        headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"},
    )
    
    with urllib.request.urlopen(req, context=ssl.create_default_context()) as resp:
        soup = BeautifulSoup(resp.read().decode("utf-8", errors="replace"), "html.parser")
    
    words = [
        tds[0].get_text(strip=True)
        for tr in soup.find_all("tr")
        if len(tds := tr.find_all("td")) >= 2
        and re.fullmatch(r"[A-Z]{5}", tds[0].get_text(strip=True))
        and re.fullmatch(r"\d+(?:\.\d+)?%", tds[1].get_text(strip=True))
    ]
    
    return(words)
    
if __name__ == "__main__":
    words = get_NYT_answer_words()
    print(f"Found {len(words)} words")
    with open("NYT_wordlist.txt", "w") as f:
        f.write("\n".join(word.lower() for word in words))
