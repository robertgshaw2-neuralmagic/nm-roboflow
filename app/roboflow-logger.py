from deepsparse.loggers import BaseLogger, MetricCategories
from typing import Any, Optional
from PIL import Image
import io, requests, datetime
from requests_toolbelt.multipart.encoder import MultipartEncoder

def save_as_img(pipeline_inputs):
    img = Image.fromarray(pipeline_inputs.images[0], mode="RGB")
    buffered = io.BytesIO()
    img.save(buffered, quality=90, format="JPEG")
    img_name = f"production-image-{datetime.datetime.now()}.jpg"
    return MultipartEncoder(fields={'file': (img_name, buffered.getvalue(), "image/jpeg")})

class RoboflowLogger(BaseLogger):
    def __init__(self, dataset_name: str, api_key: str):
        self.upload_url = f"https://api.roboflow.com/dataset/{dataset_name}/upload?api_key={api_key}"
        super(RoboflowLogger, self).__init__()

    def log(self, identifier: str, value: Any, category: Optional[str]=None):
        if "pipeline_inputs__save_as_img" in identifier:
            r = requests.post(self.upload_url, data=value, headers={'Content-Type': value.content_type})