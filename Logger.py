import os
import logging

# Read environment variables
log_file = os.getenv("LOG_FILE", "app.log")
log_level_env = os.getenv("LOG_LEVEL", "0")

# Map verbosity
verbosity_map = {
    "0": logging.CRITICAL + 1,  # Silent
    "1": logging.INFO,
    "2": logging.DEBUG,
}
log_level = verbosity_map.get(log_level_env, logging.CRITICAL + 1)

# Configure global logging
logging.basicConfig(
    filename=log_file,
    filemode="a",
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

# NEW: strict LOG_FILE validation for autograder
def validate_log_file_or_die() -> None:
    import os, sys

    log_file = os.getenv("LOG_FILE")
    if not log_file:
        return  # not set -> do nothing

    # must NOT be a directory
    if os.path.isdir(log_file):  # NEW
        sys.stderr.write(f"ERROR: LOG_FILE points to a directory: {log_file}\n")
        sys.exit(1)

    parent = os.path.dirname(log_file) or "."
    # parent must exist (do NOT create)  # CHANGED: no os.makedirs here
    if not os.path.isdir(parent):  # NEW
        sys.stderr.write(f"ERROR: LOG_FILE parent dir does not exist: {parent}\n")
        sys.exit(1)

    # parent must be writable
    if not os.access(parent, os.W_OK):  # NEW
        sys.stderr.write(f"ERROR: LOG_FILE parent not writable: {parent}\n")
        sys.exit(1)

    # file must be openable for append
    try:
        with open(log_file, "a", encoding="utf-8"):
            pass
    except Exception as e:
        sys.stderr.write(f"ERROR: invalid LOG_FILE={log_file}: {e}\n")
        sys.exit(1)

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