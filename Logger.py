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

    raw = os.getenv("LOG_FILE")
    if raw is None:
        return  # not set -> do nothing

    # Normalize and validate non-empty
    path = raw.strip()
    if not path:
        sys.stderr.write("ERROR: LOG_FILE is empty or whitespace.\n")
        sys.exit(1)

    # Expand ~ and env vars for robustness
    path = os.path.expanduser(os.path.expandvars(path))

    # Must NOT be a directory (catch trailing separators and actual dirs)
    if path.endswith(os.sep) or (os.altsep and path.endswith(os.altsep)) or os.path.isdir(path):
        sys.stderr.write(f"ERROR: LOG_FILE points to a directory: {raw}\n")
        sys.exit(1)

    parent = os.path.dirname(path) or "."
    # Parent must exist (do NOT create it)
    if not os.path.isdir(parent):
        sys.stderr.write(f"ERROR: LOG_FILE parent dir does not exist: {parent}\n")
        sys.exit(1)

    # Parent should be writable; final check is an actual open() attempt
    if not os.access(parent, os.W_OK):
        sys.stderr.write(f"ERROR: LOG_FILE parent not writable: {parent}\n")
        sys.exit(1)

    # File must be openable for append (create if missing)
    try:
        with open(path, "a", encoding="utf-8"):
            pass
    except Exception as e:
        sys.stderr.write(f"ERROR: invalid LOG_FILE={path}: {e}\n")
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