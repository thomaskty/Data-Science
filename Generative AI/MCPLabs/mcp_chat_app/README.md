# MCP Chat App

A minimal CLI-based chat application that integrates MCP concepts with both a custom TCP server and a public MQTT server.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Custom MCP Server
```bash
python3 custom_mcp_server.py --host 127.0.0.1 --port 8888
```

## Run Clients

### TCP mode (custom server)
```bash
python3 mcp_chat_client.py tcp --host 127.0.0.1 --port 8888 --name alice
python3 mcp_chat_client.py tcp --host 127.0.0.1 --port 8888 --name bob
```

### MQTT mode (public broker)
```bash
python3 mcp_chat_client.py mqtt --broker test.mosquitto.org --topic mcp/chat/test --name carol
python3 mcp_chat_client.py mqtt --broker test.mosquitto.org --topic mcp/chat/test --name dave
```
