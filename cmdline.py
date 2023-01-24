from shellcolors import ShellColors

def input_boolean_prompt(prompt_str, true="y", false="n", default=False):
    """Get user input from command line."""

    while True:
        answer = input(ShellColors.BOLD + prompt_str +
                       " " + ShellColors.ENDC).lower()
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
