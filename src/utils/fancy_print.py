import colorama
from colorama import Fore, Style

def fancy_print(message, color=Fore.WHITE, style=Style.NORMAL):
    print(f"{style}{color}{message}{Style.RESET_ALL}")