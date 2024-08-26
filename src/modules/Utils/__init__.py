import os
import sys
import configparser
from colorama import Fore, Style


def get_config_files(key: str):
    config_path = os.path.join(os.path.abspath(os.path.curdir), 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_path)
    return config[key]


def print_progress(epoch, num_epochs, batch, num_batches, loss, bar_length=30):
    progress = (batch + 1) / num_batches
    progress_bar = int(progress * bar_length)

    sys.stdout.write(
        f"{Fore.CYAN}{Style.BRIGHT}Epoch [{epoch}/{num_epochs}] |")
    sys.stdout.write(f"{Fore.YELLOW}{
                     '█' * progress_bar}{Fore.WHITE}{'-' * (bar_length - progress_bar)}")
    sys.stdout.write(f"| {Fore.GREEN}{int(progress * 100)}% {Style.RESET_ALL}")
    sys.stdout.write(
        f" [Batch {batch + 1}/{num_batches}] [Loss: {Fore.RED}{loss:.4f}{Style.RESET_ALL}]\r")
    sys.stdout.flush()


__all__ = ['print_progress', 'get_config_files']
