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


ENABLE_COLORS = False


def print_bold(msg):
    if ENABLE_COLORS:
        print(ShellColors.BOLD + str(msg) + ShellColors.ENDC)
    else:
        print(msg)


def print_ok(msg):
    if ENABLE_COLORS:
        print(ShellColors.OKGREEN + str(msg) + ShellColors.ENDC)
    else:
        print(msg)


def print_warn(msg):
    if ENABLE_COLORS:
        print(ShellColors.WARNING + "WARNING:" +
              str(msg) + ShellColors.ENDC, file=sys.stderr)
    else:
        print(msg)


def print_fail(msg):
    if ENABLE_COLORS:
        print(ShellColors.FAIL + "ERROR:" + str(msg) +
              ShellColors.ENDC, file=sys.stderr)
    else:
        print(msg)
