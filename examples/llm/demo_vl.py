import argparse
from ollama import chat, ChatResponse

def generate_response(prompt_text: str, image_path: str = None):
    # Construct user message
    message = {
        'role': 'user',
        'content': prompt_text,
    }
    
    # If an image path is provided, add it to the images field
    if image_path:
        message['images'] = [image_path]

    try:
        # Call the ollama chat API using the qwen3.5 model
        response: ChatResponse = chat(model='qwen3.5', messages=[message])
        print("\nResponse:\n" + response.message.content)
    except Exception as e:
        print(f"\nRequest failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama qwen3.5 text and image multimodal input demo")
    parser.add_argument("--text", "-t", type=str, default="Please describe this image", help="Text prompt")
    parser.add_argument("--image", "-i", type=str, default=None, help="Local image path (e.g., ./test.jpg)")
    
    args = parser.parse_args()
    
    print(f"Using model: qwen3.5")
    print(f"Text prompt: {args.text}")
    if args.image:
        print(f"Image input: {args.image}")
    
    generate_response(args.text, args.image)