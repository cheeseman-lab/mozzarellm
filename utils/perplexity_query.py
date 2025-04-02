import os
import logging
import time
from openai import OpenAI


def perplexity_chat(context, prompt, model, temperature, max_tokens, log_file):
    """
    Query the Perplexity API (which can access DeepSeek models) with a prompt.

    Args:
        context: System context for the model
        prompt: User prompt to send
        model: Model to use (e.g., "deepseek-coder", "llama-3-70b", "mistral-large")
        temperature: Temperature setting for generation
        max_tokens: Maximum tokens to generate
        log_file: File to log API calls

    Returns:
        analysis_text: The model's response
        error_message: Error message if any
    """
    logger = logging.getLogger(log_file)

    # Get API key from environment
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        logger.error("PERPLEXITY_API_KEY not found in environment variables")
        return None, "API key not found"

    # Initialize OpenAI client with Perplexity base URL
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

    # Set up messages
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": prompt},
    ]

    # Make API call with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Create chat completion
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract the response text
            analysis_text = response.choices[0].message.content

            # Log successful API call
            logger.info(f"Perplexity API call successful: model={model}")

            return analysis_text, None

        except Exception as e:
            error_message = f"Attempt {attempt+1}/{max_retries} failed: {str(e)}"
            logger.error(error_message)

            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                return None, error_message

    return None, "Maximum retries exceeded"
