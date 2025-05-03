# mozzarellm/utils/config_utils.py
import os
import importlib.resources
import logging


def get_config_path(config_name=None, provider="openai"):
    """Get path to a config file, automatically finding it in the package"""
    if config_name:
        filename = config_name
    else:
        filename = f"config_{provider}.json"

    # Try different locations
    possible_locations = [
        # Current directory
        filename,
        # configs subdirectory
        os.path.join("configs", filename),
        # mozzarellm/configs subdirectory
        os.path.join("mozzarellm", "configs", filename),
    ]

    # Check all possible file locations
    for location in possible_locations:
        if os.path.exists(location):
            return location

    # If not found in file system, try to get from package resources
    try:
        # Try to find it in the package
        return str(importlib.resources.files("mozzarellm") / "configs" / filename)
    except (ImportError, ModuleNotFoundError) as e:
        logging.warning(f"Could not find config file {filename}: {e}")
        return None


def get_prompt_path(prompt_name="top_targets.txt"):
    """Get path to a prompt file, automatically finding it in the package"""
    # Try different locations
    possible_locations = [
        # Current directory
        prompt_name,
        # prompts subdirectory
        os.path.join("prompts", prompt_name),
        # mozzarellm/prompts subdirectory
        os.path.join("mozzarellm", "prompts", prompt_name),
    ]

    # Check all possible file locations
    for location in possible_locations:
        if os.path.exists(location):
            return location

    # If not found in file system, try to get from package resources
    try:
        # Try to find it in the package
        return str(importlib.resources.files("mozzarellm") / "prompts" / prompt_name)
    except (ImportError, ModuleNotFoundError) as e:
        logging.warning(f"Could not find prompt file {prompt_name}: {e}")
        return None
