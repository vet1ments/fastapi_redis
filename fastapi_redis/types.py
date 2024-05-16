from typing import NewType
from redis import (
    Redis as Redis_,
)
from redis.asyncio import (
    Redis as AsyncRedis_,
)

RedisClient = NewType("RedisClient", Redis_)
AsyncRedisClient = NewType("AsyncRedisClient", AsyncRedis_)

