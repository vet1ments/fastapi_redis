from fastapi_redis import FastAPIRedis, FastAPIAsyncRedis, CacheClient
from fastapi_redis.types import AsyncRedisClient
from os import getpid
from fastapi import FastAPI, Depends, Form, APIRouter

app = FastAPI()
redis = FastAPIRedis()
from pydantic import BaseModel

class Item(BaseModel):
    name: str

print(3)
@app.post("/")
async def post(
        a: str,
        cache: CacheClient = Depends(redis.cache(ex=50))
):
    if not cache.is_empty:
        return cache.value
    print(cache.value)

    cache.save({"name": "1234", "q": "5678"})
    cache.save(

    )
    return cache.value


from uvicorn import run
if __name__ == "__main__":
    run(
        # "test:app",
        app=app,
        port=14457,
        )