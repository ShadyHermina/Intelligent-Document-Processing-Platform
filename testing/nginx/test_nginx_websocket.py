# test_nginx_websocket.py
# Verifies that a WebSocket connection can be established through Nginx
# to the FastAPI /ws/echo endpoint.
# Sends a message, expects the same message echoed back.
#
# Uses Python's built-in socket module to perform the WebSocket handshake
# manually — no external libraries needed.

import socket
import hashlib
import base64
import os

def websocket_test(host, port, path):
    # Generate a random WebSocket key — required by the WS handshake spec
    raw_key = os.urandom(16)
    ws_key = base64.b64encode(raw_key).decode()

    # Open a raw TCP connection to Nginx
    sock = socket.create_connection((host, port), timeout=10)

    # Send the HTTP upgrade request — this is how WebSocket connections start
    handshake = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {ws_key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"\r\n"
    )
    sock.sendall(handshake.encode())

    # Read the server's response to the upgrade request
    response = b""
    while b"\r\n\r\n" not in response:
        response += sock.recv(1024)

    response_text = response.decode(errors="replace")
    print("Handshake response:")
    print(response_text)

    assert "101 Switching Protocols" in response_text, (
        f"Expected 101 Switching Protocols, got:\n{response_text}"
    )
    print("WebSocket handshake successful — connection upgraded\n")

    # Send a WebSocket text frame containing "hello"
    # WebSocket frames have a specific binary format:
    #   byte 1: FIN bit + opcode (0x81 = final frame, text)
    #   byte 2: MASK bit + payload length (0x85 = masked, length 5)
    #   bytes 3-6: masking key (4 random bytes)
    #   bytes 7+: payload XORed with the masking key
    message = b"hello"
    mask = os.urandom(4)
    masked_payload = bytes(message[i] ^ mask[i % 4] for i in range(len(message)))
    frame = bytes([0x81, 0x80 | len(message)]) + mask + masked_payload
    sock.sendall(frame)
    print(f"Sent: hello")

    # Read the echo response frame from FastAPI
    response_frame = sock.recv(1024)
    # Parse the response frame:
    #   byte 0: FIN + opcode
    #   byte 1: payload length (no mask on server responses)
    #   bytes 2+: payload
    payload_len = response_frame[1] & 0x7F
    received = response_frame[2:2 + payload_len].decode()
    print(f"Received: {received}")

    assert received == "echo: hello", (
        f"Expected 'echo: hello', got '{received}'"
    )

    sock.close()
    return True

print("Testing WebSocket connection through Nginx at /ws/echo\n")

success = websocket_test("nginx", 80, "/ws/echo")

print("\nPASS — WebSocket connection through Nginx is working correctly")