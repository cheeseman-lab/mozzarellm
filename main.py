# Add these imports at the top
from utils.anthropic_query import anthropic_chat
from utils.perplexity_query import perplexity_chat

# Then modify the main function to handle these new model types
# Inside your main() function, replace the model selection logic with:

try:
    prompt = make_user_prompt_with_score(genes)
    finger_print = None
    
    if model.startswith('gpt'):
        print("Accessing OpenAI API")
        analysis, finger_print = openai_chat(context, prompt, model, temperature, max_tokens, rate_per_token, LOG_FILE, DOLLAR_LIMIT, seed)
    elif model.startswith('gemini'):
        print("Using Google Gemini API")
        analysis, error_message = query_genai_model(f"{context}\n{prompt}", model, temperature, max_tokens, LOG_FILE) 
    elif model.startswith('claude'):
        print("Using Anthropic Claude API")
        analysis, error_message = anthropic_chat(context, prompt, model, temperature, max_tokens, LOG_FILE, seed)
    elif model.startswith('deepseek') or model.startswith('llama') or model.startswith('mistral'):
        print(f"Using Perplexity API with {model}")
        analysis, error_message = perplexity_chat(context, prompt, model, temperature, max_tokens, LOG_FILE)
    else:
        print("Using server model")
        analysis, error_message = server_model_chat(context, prompt, model, temperature, max_tokens, LOG_FILE, seed)