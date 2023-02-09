import glob
import os

from ffmpeg_sync.index_xlsx import validate_xlsx


def glob_index_files(basepath):
    index_file_pattern = os.path.join(basepath, "**", "*_indices.xlsx")
    glob_paths = glob.glob(index_file_pattern, recursive=True)
    print(index_file_pattern, glob_paths)

    if glob_paths:
        print("\nValidating index files:")

    paths = []
    for path in glob_paths:
        print("  %s" % path)
        validation_msgs = validate_xlsx(path)
        if not validation_msgs:
            paths.append(path)
        else:
            for msg in validation_msgs:
                if msg.find("Cannot open") >= 0 and msg.find("~"):
                    # case Excel temp file -> ignore warning
                    pass
                else:
                    print("WARNING:", msg)
    return paths
