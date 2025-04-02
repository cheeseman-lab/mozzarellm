import anthropic
import logging
import time
import os

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
        error_message: Error message if any
    """
    logger = logging.getLogger(log_file)
    
    # Get API key from environment if not already configured
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in environment variables")
        return None, "API key not found"
    
    # Initialize the Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Make API call with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Create the message request
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=context,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            # Extract the analysis
            # The response content is a list of MessageContent objects
            analysis_text = ""
            for content in response.content:
                if content.type == "text":
                    analysis_text += content.text
            
            # Log successful API call
            logger.info(f"Anthropic API call successful: model={model}")
            
            return analysis_text, None
            
        except Exception as e:
            error_message = f"Attempt {attempt+1}/{max_retries} failed: {str(e)}"
            logger.error(error_message)
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None, error_message
    
    return None, "Maximum retries exceeded"