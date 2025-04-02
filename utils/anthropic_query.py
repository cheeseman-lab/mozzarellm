import anthropic
import logging
import time

def anthropic_chat(context, prompt, model, temperature, max_tokens, log_file, seed=None):
    """
    Query the Anthropic Claude API with a prompt.
    
    Args:
        context: System context for the model
        prompt: User prompt to send
        model: Anthropic model to use (e.g., "claude-3-7-sonnet-20250219")
        temperature: Temperature setting for generation
        max_tokens: Maximum tokens to generate
        log_file: File to log API calls
        seed: Random seed for reproducibility
        
    Returns:
        analysis_text: The model's response
    """
    logger = logging.getLogger(log_file)
    
    # Initialize the Anthropic client
    # API key is loaded from ANTHROPIC_API_KEY environment variable by default
    client = anthropic.Anthropic()
    
    # Make API call with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Create the message
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=context,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract the analysis
            analysis_text = response.content[0].text
            
            # Log successful API call
            logger.info(f"Anthropic API call successful: model={model}")
            
            return analysis_text, None
            
        except Exception as e:
            logger.error(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    return None, None