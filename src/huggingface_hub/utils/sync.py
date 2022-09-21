import asyncio


# https://stackoverflow.com/questions/63587660/yielding-asyncio-generator-data-back-from-event-loop-possible/63595496#63595496
def iter_over_async(ait):
    loop = asyncio.get_event_loop()
    ait = ait.__aiter__()

    async def get_next():
        try:
            obj = await ait.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None

    while True:
        done, obj = loop.run_until_complete(get_next())
        if done:
            break
        yield obj
