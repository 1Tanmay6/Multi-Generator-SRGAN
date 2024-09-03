import os
import torch
import time
from PIL import Image
import PIL.PngImagePlugin
from torchvision import transforms
from torchvision.utils import save_image

from typing import Union


class Inferencer():
    def __init__(self, model_path: str) -> None:
        self.model = None
        self.model_path = model_path
        self.cur_image = None

    def __call__(self, image_path: str) -> torch.Tensor:
        return self.infer(image_path)

    def __del__(self) -> None:
        if self.model is not None:
            del self.model

    def __repr__(self) -> str:
        return f'''Inferencer(\nMODEL_PATH: {self.model_path})'''

    def __str__(self) -> str:
        model_name = self.model_path.split('/')[-1]
        return f'''Inferencer(\nMODEL_NAME: {model_name}\nMODEL_PATH: {self.model_path}\n)'''

    def get_model(self) -> Union[str, torch.jit._script.RecursiveScriptModule]:
        if self.model is not None:
            return self.model
        return 'No Model Found'

    def load_model(self) -> Union[str, None]:
        if self.model_path is not None:
            self.model = torch.jit.load(self.model_path)
            self.model.eval()
            return 'Model loaded Successfully.'
        else:
            return 'Model Path not Found or is not valid.'

    def infer(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path)
        st = time.time()
        if self.model is None:
            self.load_model(self.model_path)
        image = self.transform_image(image)
        with torch.no_grad():
            output = self.model(image)
        print('Response Time:', time.time() - st)
        self.cur_image = output.data
        return output.data

    def transform_image(self, image: PIL.PngImagePlugin.PngImageFile) -> torch.Tensor:
        preprocess = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return preprocess(image).unsqueeze(0).cuda()

    def save(self, dir_path: str, image_name: str) -> Union[str, None]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        if self.cur_image is None:
            return 'Uable to process cur_image'
        else:
            path = os.path.join(dir_path, image_name)
            save_image(self.cur_image, path, normalize=True)
            print(f'Image saved at location: {path}')
