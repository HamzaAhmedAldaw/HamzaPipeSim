import requests

OWNER = 'HamzaAhmedAldaw'
REPO = 'HamzaPipeSim'
BRANCH = 'main'
API_URL = f'https://api.github.com/repos/{OWNER}/{REPO}/contents'

def get_contents(path=''):
    params = {'ref': BRANCH, 'per_page': 100}
    resp = requests.get(f"{API_URL}/{path}" if path else API_URL, params=params)
    resp.raise_for_status()
    return resp.json()

def crawl(path='', urls=None):
    if urls is None:
        urls = []
    for entry in get_contents(path):
        if entry['type'] == 'dir':
            crawl(entry['path'], urls)
        elif entry['type'] == 'file':
            raw = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{entry['path']}"
            urls.append(raw)
    return urls

if __name__ == '__main__':
    all_urls = crawl()
    for u in all_urls:
        print(u)