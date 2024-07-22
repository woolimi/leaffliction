RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'


def warning(text: str):
    return f"{YELLOW}{text}{END}"


def danger(text: str):
    return f"{RED}{text}{END}"


def success(text: str):
    return f"{GREEN}{text}{END}"


def info(text: str):
    return f"{BLUE}{text}{END}"
