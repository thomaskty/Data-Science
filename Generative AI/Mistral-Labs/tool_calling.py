import os
import json
from pathlib import Path
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
# Initialize Mistral client
client = Mistral(api_key=api_key)

# Define the tool for creating directories
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

def run_agent(user_prompt: str):
    """
    Run the Mistral agent with tool calling capability.
    
    Args:
        user_prompt: The user's natural language request
    """
    print(f"\n{'='*60}")
    print(f"User Request: {user_prompt}")
    print(f"{'='*60}\n")
    
    # Add context about desktop location
    desktop_path = get_desktop_path()
    enhanced_prompt = f"{user_prompt}\n\nNote: The desktop path is: {desktop_path}"
    
    messages = [
        {
            "role": "user",
            "content": enhanced_prompt
        }
    ]
    
    # First API call - LLM decides which tool to use
    response = client.chat.complete(
        model="mistral-large-latest",  # or "mistral-small-latest" for faster responses
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    print("LLM Response:")
    print(f"Stop Reason: {response.choices[0].finish_reason}\n")
    
    # Check if LLM wants to call a tool
    if response.choices[0].finish_reason == "tool_calls":
        tool_calls = response.choices[0].message.tool_calls
        
        # Add assistant's response to messages
        messages.append(response.choices[0].message)
        
        # Process each tool call
        for tool_call in tool_calls:
            print(f"Tool Called: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}\n")
            
            # Parse arguments
            args = json.loads(tool_call.function.arguments)
            
            # Execute the tool
            if tool_call.function.name == "create_directory":
                result = create_directory(args["path"], args["directory_name"])
                
                print(f"Tool Execution Result:")
                print(f"  Success: {result['success']}")
                print(f"  Message: {result['message']}\n")
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": json.dumps(result),
                    "tool_call_id": tool_call.id
                })
        
        # Second API call - LLM processes tool results and responds to user
        final_response = client.chat.complete(
            model="mistral-large-latest",
            messages=messages
        )
        
        print(f"Final Response to User:")
        print(f"{final_response.choices[0].message.content}\n")
        
    else:
        # No tool was called
        print(f"LLM Response: {response.choices[0].message.content}\n")

# Example usage
if __name__ == "__main__":
    # Example prompts
    prompts = [
        "create a directory for storing the testing results in my pc desktop location",
        "make a folder called 'ProjectData' on my desktop",
        "I need a directory named 'backup_2024' on the desktop"
    ]
    
    # You can test with any prompt
    user_input = prompts[0]
    
    # Or use interactive mode
    # user_input = input("Enter your request: ")
    
    run_agent(user_input)