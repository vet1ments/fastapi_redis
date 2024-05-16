## FASTAPI REDIS

### USAGE
```python
from fastapi_redis import (
    FastAPIRedis,
    FastAPIAsyncRedis
)
from fastapi_redis.types import (
    RedisClient,
    AsyncRedisClient,
    FastAPIAsyncRedis,
    FastAPIRedis
) 
from fastapi import (
    FastAPI,
    Depends
)

app = FastAPI()

redis = FastAPIRedis(
    host="127.0.0.1",
    port="6379"
)

async_redis = FastAPIAsyncRedis(
    host="127.0.0.1",
    port="6379"
)

# In Depends
@app.get("/")
def get(
    rd: RedisClient = Depends(redis.get_connection)  
):
    rd.get("test")
    
    
@app.get('/')
async def get(
    rd: AsyncRedisClient = Depends(async_redis.get_connection)
):
    await rd.get("test")

    
# Outside of Depends
@app.get("/")
def get():
    conn: RedisClient
    with redis.get_connection_() as conn:
        conn.get("test")
    
@app.get("/")
async def get():
    conn: AsyncRedisClient
    async with async_redis.get_connection_() as conn:
        await conn.get("test")
```