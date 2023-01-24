import sys

class ShellColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_bold(msg):
    print(ShellColors.BOLD + str(msg) + ShellColors.ENDC)

def print_ok(msg):
    print(ShellColors.OKGREEN + str(msg) + ShellColors.ENDC)

def print_warn(msg):
    print(ShellColors.WARNING + "WARNING:" + str(msg) + ShellColors.ENDC, file=sys.stderr)

def print_fail(msg):
    print(ShellColors.FAIL + "ERROR:" + str(msg) + ShellColors.ENDC, file=sys.stderr)
