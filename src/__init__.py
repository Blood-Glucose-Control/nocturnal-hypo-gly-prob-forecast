import logging


# Configure logging globally for the entire codebase
def setup_global_logging():
    """Configure logging for the entire codebase."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
            handlers=[
                logging.StreamHandler(),  # Output to console
            ],
        )


# Auto-setup when the package is imported
setup_global_logging()
