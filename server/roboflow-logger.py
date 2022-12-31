from deepsparse.loggers import BaseLogger, MetricCategories
from typing import Any, Optional
from PIL import Image
import io, requests, datetime
from requests_toolbelt.multipart.encoder import MultipartEncoder

class RoboflowLogger(BaseLogger):
    def __init__(self, dataset_name: str, api_key: str):
        self.upload_url = f"https://api.roboflow.com/dataset/{dataset_name}/upload?api_key={api_key}"
        super(RoboflowLogger, self).__init__()

    def log(self, identifier: str, value: Any, category: Optional[str]=None):
        if category == MetricCategories.DATA:
            # unpacks value and converts to image in a buffer          
            img = Image.fromarray(value.images[0], mode="RGB")
            buffered = io.BytesIO()
            img.save(buffered, quality=90, format="JPEG")
            
            # packs as multipart
            img_name = f"production-image-{datetime.datetime.now()}.jpg"
            m =  MultipartEncoder(fields={'file': (img_name, buffered.getvalue(), "image/jpeg")})

            # uploads to roboflow
            r = requests.post(self.upload_url, data=m, headers={'Content-Type': m.content_type})