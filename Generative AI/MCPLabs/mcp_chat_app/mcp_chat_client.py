import asyncio
import json
import argparse
import sys

import paho.mqtt.client as mqtt

async def tcp_client(host, port, name):
    reader, writer = await asyncio.open_connection(host, port)
    print(f"Connected to TCP {host}:{port} as {name}")

    async def send_stdin():
        loop = asyncio.get_event_loop()
        while True:
            msg = await loop.run_in_executor(None, sys.stdin.readline)
            if not msg:
                break
            payload = {"type": "chat", "name": name, "msg": msg.strip()}
            writer.write((json.dumps(payload) + "\n").encode())
            await writer.drain()

    async def recv_server():
        while True:
            data = await reader.readline()
            if not data:
                break
            payload = json.loads(data.decode().strip())
            print(f"{payload.get('from')}: {payload.get('msg', payload)}")

    await asyncio.gather(send_stdin(), recv_server())

def mqtt_client(broker, topic, name):
    client = mqtt.Client()

    def on_connect(client, userdata, flags, rc):
        print(f"Connected to MQTT {broker}, subscribed to {topic} as {name}")
        client.subscribe(topic)

    def on_message(client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        sender = payload.get("name", "unknown")
        print(f"{sender}: {payload.get('msg')}")

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(broker, 1883, 60)

    import threading
    def input_thread():
        for line in sys.stdin:
            payload = {"name": name, "msg": line.strip()}
            client.publish(topic, json.dumps(payload))

    threading.Thread(target=input_thread, daemon=True).start()
    client.loop_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    tcp_parser = subparsers.add_parser("tcp")
    tcp_parser.add_argument("--host", default="127.0.0.1")
    tcp_parser.add_argument("--port", type=int, default=8888)
    tcp_parser.add_argument("--name", required=True)

    mqtt_parser = subparsers.add_parser("mqtt")
    mqtt_parser.add_argument("--broker", default="test.mosquitto.org")
    mqtt_parser.add_argument("--topic", default="mcp/chat/test")
    mqtt_parser.add_argument("--name", required=True)

    args = parser.parse_args()

    if args.mode == "tcp":
        asyncio.run(tcp_client(args.host, args.port, args.name))
    elif args.mode == "mqtt":
        mqtt_client(args.broker, args.topic, args.name)
