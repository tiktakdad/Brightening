import base64
from pathlib import Path

import requests

with Path("/data/venus/images/test/img_000072.jpg").open("rb") as f:
    imgstr = base64.b64encode(f.read()).decode("UTF-8")

body = {"session": "UUID", "payload": {"img": {"data": imgstr}}}
resp = requests.post("http://192.168.0.2:8000/detect", json=body)
print(resp.json())