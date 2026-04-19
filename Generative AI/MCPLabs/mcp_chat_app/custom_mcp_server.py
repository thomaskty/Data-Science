import asyncio
import json
import argparse

clients = set()

async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    clients.add(writer)
    try:
        while True:
            data = await reader.readline()
            if not data:
                break
            message = data.decode().strip()
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue
            response = await process_message(payload, addr)
            if response:
                for client in clients:
                    client.write((json.dumps(response) + "\n").encode())
                    await client.drain()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        clients.remove(writer)
        writer.close()
        await writer.wait_closed()

async def process_message(payload, addr):
    if payload.get("type") == "chat":
        return {"from": addr, "msg": payload.get("msg")}
    elif payload.get("type") == "fs:list":
        import os
        path = payload.get("path", ".")
        try:
            files = os.listdir(path)
            return {"from": "server", "files": files}
        except Exception as e:
            return {"from": "server", "error": str(e)}
    return None

async def main(host, port):
    server = await asyncio.start_server(handle_client, host, port)
    addr = server.sockets[0].getsockname()
    print(f"Serving on {addr}")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    asyncio.run(main(args.host, args.port))
