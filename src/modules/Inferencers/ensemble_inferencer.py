import os
import torch
import time
from PIL import Image
from torchvision.utils import save_image

from typing import Union

from .inferencer import Inferencer


class EnsembleInferencer(Inferencer):
    def __init__(self, model_paths: list) -> None:
        self.models = []
        self.model_paths = model_paths
        self.cur_image = None
        self.load_models()

    def __repr__(self) -> str:
        return f'''EnsembleInferencer(\nMODEL_PATHS: {self.model_paths})'''

    def __str__(self) -> str:
        model_names = [path.split('/')[-1] for path in self.model_paths]
        return f'''EnsembleInferencer(\nMODEL_NAMES: {model_names}\nMODEL_PATHS: {self.model_paths}\n)'''
    
    def __del__(self) -> None:
        for model in self.models:
            if model is not None:
                del model

    def load_models(self) -> None:
        for model_path in self.model_paths:
            model = torch.jit.load(model_path)
            model.eval()
            self.models.append(model)

    def infer(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path)
        st = time.time()
        image = self.transform_image(image)
        outputs = []
        for model in self.models:
            with torch.no_grad():
                output = model(image)
            outputs.append(output)
        output = torch.mean(torch.stack(outputs), dim=0) # Chnage this to the desired ensemble method (Currently Mean)
        print('Response Time:', time.time() - st)
        self.cur_image = output.data
        return output.data

    def save(self, dir_path: str, image_name: str) -> Union[str, None]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        if self.cur_image is None:
            return 'Uable to process cur_image'
        else:
            path = os.path.join(dir_path, image_name)
            save_image(self.cur_image, path, normalize=True)
            print(f'Image saved at location: {path}')