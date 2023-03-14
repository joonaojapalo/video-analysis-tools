
def progress(n, total):
    """Print updateable command-line progress indicator (percent)
    """
    percent_ready = (100 * (n + 1)) // total
    print("\b\b\b\b\b%3d %%" % percent_ready, end="", flush=True)


def complete():
    """Print completed command-line progress indicator
    """
    print("\b\b\b\b\b100 %", flush=True)
