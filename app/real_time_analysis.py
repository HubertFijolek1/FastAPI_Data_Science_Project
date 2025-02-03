import asyncio
import random

async def stream_data():
    """
    Yields random integers (0-100) every second as a simulation.
    """
    while True:
        yield random.randint(0, 100)
        await asyncio.sleep(1)