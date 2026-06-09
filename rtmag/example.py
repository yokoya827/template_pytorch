from rtmag.process.download.find_harp import find_ar
from PIL import Image
import requests
from io import BytesIO

d = "2024-05-09 02:00:00"
ar, img_url = find_ar(d, show_image=True)  # img_url は URL（文字列）

# 画像をダウンロードして開く
response = requests.get(img_url)
image = Image.open(BytesIO(response.content))
image.show()  # GUIがあれば表示できる。なければ image.save("out.png") に変更可

print(ar)
