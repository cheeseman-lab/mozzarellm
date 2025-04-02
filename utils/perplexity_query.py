import requests
import logging
import time
import os

def perplexity_chat(context, prompt, model, temperature, max_tokens, log_file):
    """
    Query the Perplexity API (which can access DeepSeek models) with a prompt.
    
    Args:
        context: System context for the model
        prompt: User prompt to send
        model: Model to use (e.g., "deepseek-coder")
        temperature: Temperature setting for generation
        max_tokens: Maximum tokens to generate
        log_file: File to log API calls
        
    Returns:
        analysis_text: The model's response
    """
    logger = logging.getLogger(log_file)
    
    # Get API key from environment
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        logger.error("PERPLEXITY_API_KEY not found in environment variables")
        return None, "API key not found"
    
    # Set up the API endpoint
    url = "https://api.perplexity.ai/chat/completions"
    
    # Set up headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Set up the request data
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # Make API call with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            
            # Check if the request was successful
            if response.status_code == 200:
                response_json = response.json()
                analysis_text = response_json['choices'][0]['message']['content']
                
                # Log successful API call
                logger.info(f"Perplexity API call successful: model={model}")
                
                return analysis_text, None
            else:
                error_message = f"API returned status code {response.status_code}: {response.text}"
                logger.error(error_message)
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return None, error_message
                
        except Exception as e:
            logger.error(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None, str(e)
    
    return None, "Maximum retries exceeded"