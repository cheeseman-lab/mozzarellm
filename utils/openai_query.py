import os
from openai import OpenAI
import time
import logging

def openai_chat(context, prompt, model, temperature, max_tokens, rate_per_token, log_file, dollar_limit, seed=None):
    """
    Query the OpenAI API with a prompt.
    
    Args:
        context: System context for the model
        prompt: User prompt to send
        model: OpenAI model to use
        temperature: Temperature setting for generation
        max_tokens: Maximum tokens to generate
        rate_per_token: Cost per token
        log_file: File to log API calls
        dollar_limit: Maximum cost allowed
        seed: Random seed for reproducibility
        
    Returns:
        analysis_text: The model's response
        fingerprint: The response's fingerprint for tracking
    """
    logger = logging.getLogger(log_file)
    
    # Get API key from environment if not configured
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return None, None
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Set up messages
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": prompt}
    ]
    
    # Calculate token estimate (rough approximation)
    total_chars = len(context) + len(prompt)
    estimated_tokens = total_chars / 4  # Rough estimate: ~4 chars per token
    estimated_cost = estimated_tokens * rate_per_token
    
    if estimated_cost > dollar_limit:
        logger.warning(f"Estimated cost ${estimated_cost:.4f} exceeds limit ${dollar_limit}")
        return None, None
    
    # Make API call with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed
            )
            
            # Extract the analysis and fingerprint
            analysis_text = response.choices[0].message.content
            fingerprint = getattr(response, 'system_fingerprint', None)
            
            # Log successful API call
            tokens_used = response.usage.total_tokens
            cost = tokens_used * rate_per_token
            logger.info(f"API call successful: {tokens_used} tokens, ${cost:.4f}")
            
            return analysis_text, fingerprint
            
        except Exception as e:
            error_message = f"Attempt {attempt+1}/{max_retries} failed: {str(e)}"
            logger.error(error_message)
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None, f"API error: {str(e)}"
    
    return None, "Maximum retries exceeded"