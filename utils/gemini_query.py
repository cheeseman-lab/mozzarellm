import google.generativeai as genai
import logging
import time
import os

def query_genai_model(context, prompt, model, temperature, max_tokens, log_file):
    """
    Query the Google Gemini API with a prompt.
    
    Args:
        context: System context for the model
        prompt: User prompt to send
        model: Gemini model to use (e.g., "gemini-pro" or "gemini-1.5-pro")
        temperature: Temperature setting for generation
        max_tokens: Maximum tokens to generate
        log_file: File to log API calls
        
    Returns:
        analysis_text: The model's response
        error_message: Error message if any
    """
    logger = logging.getLogger(log_file)
    
    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables")
        return None, "API key not found"
    
    # Configure the Gemini API with your API key
    genai.configure(api_key=api_key)
    
    # Create a client
    client = genai.GenerationClient()
    
    # Set up the generation config
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "top_p": 1,
        "top_k": 32,
    }
    
    # Combine context and prompt
    # For Gemini, we'll use system_instruction for the context and contents for the prompt
    system_instruction = {"role": "system", "parts": [{"text": context}]}
    contents = {"role": "user", "parts": [{"text": prompt}]}
    
    # Make API call with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Generate content
            response = client.generate_content(
                model=model,
                contents=[contents],
                system_instruction=system_instruction,
                generation_config=generation_config
            )
            
            # Extract the response text
            analysis_text = response.text
            
            # Log successful API call
            logger.info(f"Google Gemini API call successful: model={model}")
            
            return analysis_text, None
            
        except Exception as e:
            error_message = f"Attempt {attempt+1}/{max_retries} failed: {str(e)}"
            logger.error(error_message)
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None, error_message
    
    return None, "Maximum retries exceeded"