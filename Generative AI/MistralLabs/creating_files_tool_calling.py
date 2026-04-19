import os
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Option 1: Using Mistral API (requires API key)
USE_API = True  # Set to True if you have API key

if USE_API:
    from mistralai import Mistral
else:
    # Option 2: Using local model via Ollama
    import requests

def get_desktop_path() -> str:
    """Get the desktop path for the current OS"""
    home = str(Path.home())
    
    # Windows
    if os.name == 'nt':
        desktop = os.path.join(home, 'Desktop')
    # macOS and Linux
    else:
        desktop = os.path.join(home, 'Desktop')
    
    return desktop

def create_directory(path: str, directory_name: str) -> dict:
    """
    Creates a directory at the specified location.
    
    Args:
        path: The base path where directory should be created
        directory_name: Name of the directory to create
        
    Returns:
        dict: Status of the operation
    """
    try:
        # Expand user path (handles ~)
        expanded_path = os.path.expanduser(path)
        
        # Create full path
        full_path = os.path.join(expanded_path, directory_name)
        
        # Create directory (parents=True creates intermediate directories)
        os.makedirs(full_path, exist_ok=True)
        
        return {
            "success": True,
            "message": f"Directory created successfully at: {full_path}",
            "path": full_path
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error creating directory: {str(e)}",
            "path": None
        }

def write_file(directory_path: str, file_name: str, content: str) -> dict:
    """
    Writes content to a file in the specified directory.
    
    Args:
        directory_path: The path to the directory where file should be created
        file_name: Name of the file (with extension)
        content: Content to write to the file
        
    Returns:
        dict: Status of the operation
    """
    try:
        # Expand user path (handles ~)
        expanded_path = os.path.expanduser(directory_path)
        
        # Create full file path
        full_path = os.path.join(expanded_path, file_name)
        
        # Ensure directory exists
        os.makedirs(expanded_path, exist_ok=True)
        
        # Write content to file
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "message": f"File written successfully at: {full_path}",
            "path": full_path
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error writing file: {str(e)}",
            "path": None
        }

# Define the tool schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Creates a directory at the specified path. Can create nested directories if they don't exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The full path where the directory should be created"
                    },
                    "directory_name": {
                        "type": "string",
                        "description": "The name of the directory to create"
                    }
                },
                "required": ["path", "directory_name"]
            }
        }
    }
]

def run_agent_with_ollama(user_prompt: str):
    """
    Run the agent using Ollama (local Mistral model).
    Requires Ollama to be installed with a Mistral model.
    """
    print(f"\n{'='*60}")
    print(f"User Request: {user_prompt}")
    print(f"{'='*60}\n")
    
    desktop_path = get_desktop_path()
    
    # Create system prompt with tool information
    system_prompt = f"""You are a helpful assistant that can create directories on the user's computer.

Available tools:
{json.dumps(tools, indent=2)}

When the user asks to create a directory, you should respond with a JSON object containing the tool call:
{{
    "tool": "create_directory",
    "arguments": {{
        "path": "<path>",
        "directory_name": "<name>"
    }}
}}

The user's desktop path is: {desktop_path}

If the user mentions "desktop" or "desktop location", use the path: {desktop_path}
"""

    # Call Ollama API
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "mistral",  # or "mistral-nemo", "mistral-small"
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False
            }
        )
        
        if response.status_code != 200:
            print(f"Error: Ollama returned status {response.status_code}")
            print(f"Make sure Ollama is running and Mistral model is installed.")
            print(f"Install with: ollama pull mistral")
            return
        
        llm_response = response.json()["message"]["content"]
        print(f"LLM Response:\n{llm_response}\n")
        
        # Try to extract JSON tool call from response
        try:
            # Look for JSON in the response
            start = llm_response.find('{')
            end = llm_response.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = llm_response[start:end]
                tool_call = json.loads(json_str)
                
                if tool_call.get("tool") == "create_directory":
                    args = tool_call["arguments"]
                    print(f"Tool Called: create_directory")
                    print(f"Arguments: {args}\n")
                    
                    # Execute the tool
                    result = create_directory(args["path"], args["directory_name"])
                    
                    print(f"Tool Execution Result:")
                    print(f"  Success: {result['success']}")
                    print(f"  Message: {result['message']}\n")
                    
                    # Get final response from LLM
                    final_response = requests.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": "mistral",
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                                {"role": "assistant", "content": llm_response},
                                {"role": "user", "content": f"Tool execution result: {json.dumps(result)}. Please provide a natural response to the user."}
                            ],
                            "stream": False
                        }
                    )
                    
                    final_text = final_response.json()["message"]["content"]
                    print(f"Final Response:\n{final_text}\n")
        
        except json.JSONDecodeError:
            print("Could not parse tool call from LLM response.")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama.")
        print("Make sure Ollama is running: ollama serve")
        print("And Mistral model is installed: ollama pull mistral")

def run_agent_with_api(user_prompt: str, api_key: str):
    """
    Run the agent using Mistral API.
    """
    print(f"\n{'='*60}")
    print(f"User Request: {user_prompt}")
    print(f"{'='*60}\n")
    
    client = Mistral(api_key=api_key)
    desktop_path = get_desktop_path()
    enhanced_prompt = f"{user_prompt}\n\nNote: The desktop path is: {desktop_path}"
    
    messages = [{"role": "user", "content": enhanced_prompt}]
    
    # First API call
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    print("LLM Response:")
    print(f"Stop Reason: {response.choices[0].finish_reason}\n")
    
    if response.choices[0].finish_reason == "tool_calls":
        tool_calls = response.choices[0].message.tool_calls
        messages.append(response.choices[0].message)
        
        for tool_call in tool_calls:
            print(f"Tool Called: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}\n")
            
            args = json.loads(tool_call.function.arguments)
            
            if tool_call.function.name == "create_directory":
                result = create_directory(args["path"], args["directory_name"])
                
                print(f"Tool Execution Result:")
                print(f"  Success: {result['success']}")
                print(f"  Message: {result['message']}\n")
                
                messages.append({
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": json.dumps(result),
                    "tool_call_id": tool_call.id
                })
        
        final_response = client.chat.complete(
            model="mistral-large-latest",
            messages=messages
        )
        
        print(f"Final Response:\n{final_response.choices[0].message.content}\n")
    else:
        print(f"LLM Response: {response.choices[0].message.content}\n")

if __name__ == "__main__":
    # Example prompt
    user_input = """
    create 4 essays regarding artificial intelligence and each of them consisting of 3 sentences. and store these 4 essays in a directory called essay in {cwd}. 
    Name them essay_1.txt, essay_2.txt, essay_3.txt, and essay_4.txt
    """

    
    if USE_API:
        # If using Mistral API
        api_key = os.getenv("MISTRAL_API_KEY") or "YOUR_API_KEY_HERE"
        if api_key == "YOUR_API_KEY_HERE":
            print("Error: Please set your MISTRAL_API_KEY environment variable")
            print("Or replace 'YOUR_API_KEY_HERE' with your actual API key")
        else:
            run_agent_with_api(user_input, api_key)
    else:
        # Using local Ollama (recommended for open source)
        print("Using local Ollama with Mistral model...")
        print("Make sure Ollama is installed and running!")
        print("Installation: https://ollama.ai/download")
        print("Install Mistral: ollama pull mistral\n")
        run_agent_with_ollama(user_input)