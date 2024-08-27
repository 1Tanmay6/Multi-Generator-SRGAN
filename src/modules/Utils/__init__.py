import os
import sys
import torch
import configparser
from colorama import Fore, Style

CYAN = Fore.CYAN
GRAY = Fore.LIGHTBLACK_EX
GREEN = Fore.GREEN
WHITE = Fore.WHITE
RED = Fore.RED
BLUE = Fore.LIGHTBLUE_EX


def load_model(model_path, model=None, type='script'):
    if type == 'script':
        model_scripted = torch.jit.load(model_path)
        return model_scripted
    else:
        model.load_state_dict(torch.load(model_path))
        return model


def get_config_files(key: str):
    config_path = os.path.join(os.path.abspath(os.path.curdir), 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_path)
    return config[key]


def print_progress(epoch, num_epochs, batch, num_batches, loss, bar_length=30, adversarial_loss=False):
    progress = (batch + 1) / num_batches
    progress_bar = int(bar_length * progress)

    sys.stdout.write(f"{CYAN}{Style.BRIGHT}Epoch [{epoch}/{num_epochs}] |")
    sys.stdout.write(
        f"{GRAY}{'█' * progress_bar}{WHITE}{' ' * (bar_length - progress_bar)}")
    sys.stdout.write(f"| {GREEN}{int(progress * 100)}%{Style.RESET_ALL}")

    if adversarial_loss:
        sys.stdout.write(
            f" [Batch {batch + 1}/{num_batches}] [D loss: {RED}{loss[0]:.4f}{Style.RESET_ALL}] [G loss: {BLUE}{loss[1]:.4f}{Style.RESET_ALL}]\r")
    else:
        sys.stdout.write(
            f" [Batch {batch + 1}/{num_batches}] [Loss: {RED}{loss:.4f}{Style.RESET_ALL}]\r")

    sys.stdout.flush()


__all__ = ['print_progress', 'get_config_files', 'load_model']
