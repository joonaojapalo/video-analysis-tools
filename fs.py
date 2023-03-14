import os
import glob

def globdirs(input_glob):
    """Glob only directories by input_glob string. 
    """
    globs = glob.glob(input_glob)
    return [ x for x in globs  if os.path.isdir(x)]
