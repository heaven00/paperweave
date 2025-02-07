import requests
from bs4 import BeautifulSoup


import arxiv


def get_paper_title(arxiv_id):
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    result = next(client.results(search))

    return result.title


# Example usage:


def get_paper_summary(arxiv_id: str) -> str:
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    result = next(client.results(search))

    return result.summary


def get_paper_authors(arxiv_id: str) -> str:
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    result = next(client.results(search))
    authors = result.authors
    authors = [author.name for author in authors]
    return authors


def get_arxiv_text(arxiv_code: str) -> str:
    url = f"https://arxiv.org/html/{arxiv_code}"

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)

    return text
