import asyncio
import websockets


async def main():

    uri = "ws://127.0.0.1:8000/ws/agent"

    async with websockets.connect(uri) as websocket:

        question = input("Enter your question: ")

        await websocket.send(question)

        while True:

            response = await websocket.recv()

            print("\nServer:", response)

            if '"type":"done"' in response:
                break


asyncio.run(main())