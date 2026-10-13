import requests
import re
import json

url = "https://www.instagram.com/reel/DV1YsvEilNl/embed/"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}
resp = requests.get(url, headers=headers, timeout=10)
print("Status:", resp.status_code)
# Search for video_url or mp4
matches = re.findall(r'"video_url":"([^"]+)"', resp.text)
if not matches:
    matches = re.findall(r'https[^\'"\\]+?\.mp4[^\'"\s<]*', resp.text)

print("Matches count:", len(matches))
if matches:
    clean_url = matches[0].encode().decode('unicode_escape').replace('\\', '')
    print("Clean MP4 URL:", clean_url[:120])
