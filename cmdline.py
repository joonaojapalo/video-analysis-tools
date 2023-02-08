from shellcolors import ShellColors, ENABLE_COLORS


def get_msg(msg):
    if ENABLE_COLORS:
        return ShellColors.BOLD + msg + " " + ShellColors.ENDC
    else:
        return msg


def input_boolean_prompt(prompt_str, true="y", false="n", default=False):
    """Get user input from command line."""

    while True:
        answer = input(get_msg(prompt_str)).lower()
        if default == False:
            accept = [true.lower()]
            decline = [false.lower(), '']
        else:
            accept = [true.lower(), '']
            decline = [false.lower()]

        if answer in accept:
            return True
        elif answer in decline:
            return False
