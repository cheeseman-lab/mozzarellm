import requests
import json
import logging
import time
import os


def server_model_chat(
    context, prompt, model, temperature, max_tokens, log_file, seed=None
):
    """
    Query a custom server-hosted model with a prompt.

    Args:
        context: System context for the model
        prompt: User prompt to send
        model: Model identifier (e.g., "local-llama" or "server:port/model")
        temperature: Temperature setting for generation
        max_tokens: Maximum tokens to generate
        log_file: File to log API calls
        seed: Random seed for reproducibility

    Returns:
        analysis_text: The model's response
        error_message: Error message if any
    """
    logger = logging.getLogger(log_file)

    # Get server URL from environment or construct from model identifier
    if ":" in model:
        # Assume model is in format "server:port/model_name"
        server_parts = model.split("/")
        server_url = f"http://{server_parts[0]}/v1/chat/completions"
        model_name = server_parts[1] if len(server_parts) > 1 else "default"
    else:
        # Get server URL from environment
        server_url = os.environ.get(
            "MODEL_SERVER_URL", "http://localhost:8000/v1/chat/completions"
        )
        model_name = model

    # Set up the headers
    headers = {"Content-Type": "application/json"}

    # API key for the server, if needed
    server_api_key = os.environ.get("MODEL_SERVER_API_KEY")
    if server_api_key:
        headers["Authorization"] = f"Bearer {server_api_key}"

    # Set up the request data
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Add seed if provided
    if seed is not None:
        data["seed"] = seed

    # Make API call with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                server_url, headers=headers, json=data, timeout=120
            )

            # Check if the request was successful
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    analysis_text = response_json["choices"][0]["message"]["content"]

                    # Log successful API call
                    logger.info(f"Server model API call successful: model={model_name}")

                    return analysis_text, None
                except (KeyError, json.JSONDecodeError) as e:
                    error_message = f"Failed to parse response: {str(e)}"
                    logger.error(error_message)

                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)  # Exponential backoff
                    else:
                        return None, error_message
            else:
                error_message = (
                    f"API returned status code {response.status_code}: {response.text}"
                )
                logger.error(error_message)

                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    return None, error_message

        except Exception as e:
            error_message = f"Attempt {attempt+1}/{max_retries} failed: {str(e)}"
            logger.error(error_message)

            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                return None, error_message

    return None, "Maximum retries exceeded"
