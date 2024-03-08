import requests
import base64
from PIL import Image
from io import BytesIO

url = "http://195.162.164.16:40002/generate_image"
data = {
    "prompt": "1 girl with a sword in a fantasy setting",
    "negative_prompt": "bad quality"
}

response = requests.post(url, json=data)
image_data = base64.b64decode(response.json()['image'])
image = Image.open(BytesIO(image_data))
image.show()
image.save("generated_image.png")
print(response.status_code)
print(response.json())