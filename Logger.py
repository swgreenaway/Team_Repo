import os
import logging

# NEW: strict LOG_FILE validation for autograder
def validate_log_file_or_die() -> None:
    import os, sys

    raw = os.getenv("LOG_FILE")
    if raw is None:
        return  # not set -> do nothing

    p = raw.strip()
    if not p:
        sys.stderr.write("ERROR: LOG_FILE is empty or whitespace.\n")
        sys.exit(1)

    p = os.path.expanduser(os.path.expandvars(p))

    # reject directories (incl. trailing slash) 
    if p.endswith(os.sep) or (os.altsep and p.endswith(os.altsep)) or os.path.isdir(p):
        sys.stderr.write(f"ERROR: LOG_FILE points to a directory: {raw}\n")
        sys.exit(1)

    parent = os.path.dirname(p) or "."
    if not os.path.isdir(parent):
        sys.stderr.write(f"ERROR: LOG_FILE parent dir does not exist: {parent}\n")
        sys.exit(1)

    # MUST already exist (do not create)
    if not os.path.isfile(p):
        sys.stderr.write(f"ERROR: LOG_FILE does not exist: {p}\n")
        sys.exit(1)

    # MUST be writable without creating/truncating
    try:
        with open(p, "r+", encoding="utf-8"):
            pass
    except Exception as e:
        sys.stderr.write(f"ERROR: LOG_FILE not writable: {p}: {e}\n")
        sys.exit(1)

# Read environment variables
log_file = os.getenv("LOG_FILE")  # no default; don't silently create a file
log_level_env = os.getenv("LOG_LEVEL", "0")

_level_map = {"0": logging.CRITICAL + 1, "1": logging.INFO, "2": logging.DEBUG}
log_level = _level_map.get(log_level_env, logging.CRITICAL + 1)


kwargs = dict(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

# only attach a file handler if LOG_FILE was provided
if log_file:
    kwargs.update(filename=log_file, filemode="r+")   # don't create new files

# logging.basicConfig(**kwargs)

def get_logger(name: str):
    """
    Return a logger tagged with the given module name.\n
    Best Practice: Use Function to get a logger per file.\n\t\t\t
                   "name" as file name.
    """
    return logging.getLogger(name)
    
    # Example usage:
    # from Logger import get_logger
    # logger = get_logger(__file_name__)
    # ...  