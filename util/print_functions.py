from colorama import Fore, Style, init

init(autoreset=True)


def section_print(content: str):
    bar = Fore.GREEN + "=" * (len(content) + 2) + Style.RESET_ALL
    text = Fore.GREEN + content + Style.RESET_ALL
    print(f"\n{bar}\n {text} \n{bar}\n")


if __name__ == "__main__":
    section_print("This is a section header")
    print("This is normal text after the section header.")
