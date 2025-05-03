import os


def get_default_config_path(provider="openai"):
    """Get path to default config file for the specified provider"""
    filename = f"config_{provider}.json"

    # Try current directory
    if os.path.exists(filename):
        return filename

    # Try configs directory in current directory
    local_configs = os.path.join("configs", filename)
    if os.path.exists(local_configs):
        return local_configs

    # Try installed package
    try:
        import importlib.resources

        return str(importlib.resources.files("mozzarellm") / "configs" / filename)
    except (ImportError, ModuleNotFoundError):
        # Fallback to relative path from this file
        this_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(os.path.dirname(this_dir))
        return os.path.join(package_dir, "configs", filename)


def get_default_prompt_path(prompt_name="top_targets.txt"):
    """Get path to default prompt file"""
    filename = f"{prompt_name}"

    # Try current directory
    if os.path.exists(filename):
        return filename

    # Try prompts directory in current directory
    local_prompts = os.path.join("prompts", filename)
    if os.path.exists(local_prompts):
        return local_prompts

    # Try installed package
    try:
        import importlib.resources

        return str(importlib.resources.files("mozzarellm") / "prompts" / filename)
    except (ImportError, ModuleNotFoundError):
        # Fallback to relative path from this file
        this_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(os.path.dirname(this_dir))
        return os.path.join(package_dir, "prompts", filename)
