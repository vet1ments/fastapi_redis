from redis import (
    Redis as PyRedis,
    ConnectionPool,
)
from redis.asyncio import (
    Redis as AsyncPyRedis,
    ConnectionPool as AsyncConnectionPool
)
from redis.asyncio.client import Pipeline as AsyncPipeline
from redis.client import Pipeline
from starlette.datastructures import QueryParams
from redis.exceptions import RedisError
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator, Any, Callable, Literal
from fastapi import BackgroundTasks, Depends, Request, Response
from functools import partial

from .types import (
    AsyncRedisClient,
    RedisClient
)

from inspect import iscoroutinefunction

from .utils import Singleton
from json import dumps, loads


class CacheClient:
    def __init__(
            self,
            key: str,
            response: Response,
            redis_client: RedisClient | AsyncRedisClient,
            background_tasks: BackgroundTasks,
            expire_time: int | None = None
    ):
        self._key: str = key
        self._response: Response = response
        self._value: Any | None = None
        self._background_tasks: BackgroundTasks = background_tasks
        self._expire_time: int | None = expire_time

        assert isinstance(redis_client, PyRedis) or isinstance(redis_client, AsyncPyRedis), "redis client Must be a PyRedis or AsyncRedis"

        self.init_value = partial(self._sync_init_value)
        self._save = partial(self._sync_save)

        if isinstance(redis_client, AsyncPyRedis):
            self.init_value = partial(self._async_init_value)
            self._save = partial(self._async_save)

        self._redis_client = redis_client

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(key={self.key} value={self.value})'

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def is_empty(self) -> bool:
        if self.value is None:
            return True
        return False

    @property
    def key(self) -> str:
        return self._key

    @property
    def value(self) -> Any:
        return self._value

    def _sync_init_value(self) -> None:
        if self._expire_time is None:
            return

        rd: RedisClient = self._redis_client
        value = rd.get(self._key)
        if value:
            self._value = loads(value)

    async def _async_init_value(self) -> None:
        if self._expire_time is None:
            return

        rd: AsyncRedisClient = self._redis_client
        value = await rd.get(self._key)
        if value:
            self._value = loads(value)

    def _sync_save(self) -> None:
        rd: RedisClient = self._redis_client
        key = self._key
        value = self._value

        def wrap(pipe: Pipeline) -> None:
            pipe.multi()
            pipe.set(key, dumps(value), ex=self._expire_time)
        rd.transaction(wrap, key)

    async def _async_save(self) -> None:
        rd: AsyncRedisClient = self._redis_client
        key = self._key
        value = self._value

        async def wrap(pipe: AsyncPipeline) -> None:
            pipe.multi()
            await pipe.set(key, dumps(value), ex=self._expire_time)
        await rd.transaction(wrap, key)

    def save(
            self,
            value: Any,
            expire_time: int | None = None
    ) -> None:
        expire_time = expire_time or self._expire_time
        if expire_time is None:
            return
        self._value = value
        self._background_tasks.add_task(self._save)

class BaseRedis(metaclass=Singleton):
    def __init__(
            self,
            client: Literal["sync", "async"],
            host: str | None = '127.0.0.1',
            port: str | None = '6379',
    ):
        self._host = host
        self._port = port
        self._pool = None
        assert client in ["sync", "async"], "client Must be a sync or async"
        self._client = client
        if client == "sync":
            self.get_connection_ = self._get_sync_connection_
            self.get_connection = self._get_sync_connection
        elif client == "async":
            self.get_connection_ = self._get_async_connection_
            self.get_connection = self._get_async_connection

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> str:
        return self._port

    @property
    def pool(self) -> AsyncConnectionPool | ConnectionPool | None:
        return self._pool

    @property
    def client_type(self) -> Literal["sync", "async"]:
        return self._client

    @contextmanager
    def _get_sync_connection_(self) -> Iterator[RedisClient]:
        try:
            if self._pool is None:
                self._pool = ConnectionPool(host=self._host, port=self._port, decode_responses=True)
            conn = PyRedis(
                connection_pool=self._pool,
                encoding='utf-8',
                decode_responses=True
            )
            yield conn
        except RedisError as e:
            print(e)
            raise
        finally:
            conn.close()

    def _get_sync_connection(self) -> Iterator[RedisClient]:
        """For FastAPI Depends"""
        with self.get_connection_() as conn:
            yield conn

    @asynccontextmanager
    async def _get_async_connection_(self) -> AsyncIterator[AsyncRedisClient]:
        try:
            if self._pool is None:
                self._pool = AsyncConnectionPool(host=self._host, port=self._port, decode_responses=True)
            conn = await AsyncPyRedis(
                connection_pool=self._pool,
                encoding='utf-8',
                decode_responses=True
            )
            yield conn
        except RedisError as e:
            raise e
        finally:
            await conn.close()

    async def _get_async_connection(self) -> AsyncIterator[AsyncRedisClient]:
        """For FastAPI Depends"""
        async with self.get_connection_() as conn:
            yield conn

    def background_job(self, func, background_tasks: BackgroundTasks, *args, **kwargs) -> None:
        background_tasks.add_task(
            partial(func, *args, **kwargs)
        )

    def _make_key(
            self,
            query_params: QueryParams,
            path_parms: dict
    ) -> str:
        key = ""
        for k, v in query_params.items():
            key += f"{k}={v}"

        for k, v in path_parms.items():
            key += f"{key}={v}"
        return key

    def cache(
            self,
            key: str | None = None,
            ex: int | None = None,
    ) -> Callable[[Request, RedisClient | AsyncRedisClient], AsyncIterator[CacheClient]]:
        _key = key

        async def wrap(
                request: Request,
                response: Response,
                background_tasks: BackgroundTasks,
                rd: PyRedis | AsyncPyRedis = Depends(self.get_connection),
        ) -> AsyncIterator[CacheClient]:
            key = _key
            if key is None:
                key = self._make_key(
                    query_params=request.query_params,
                    path_parms=request.path_params
                )
            cache_client = CacheClient(
                key=key,
                response=response,
                redis_client=rd,
                background_tasks=background_tasks,
                expire_time=ex
            )

            if iscoroutinefunction((func := cache_client.init_value)):
                await func()
            else:
                func()
            yield cache_client
        return wrap


class FastAPIRedis(BaseRedis):
    def __init__(
            self,
            host: str | None = '127.0.0.1',
            port: str | None = '6379'
    ):
        super().__init__(
            client="sync",
            host=host,
            port=port,
        )
        self._pool = ConnectionPool(
            host=self._host,
            port=self._port,
            decode_responses=True
        )

class FastAPIAsyncRedis(BaseRedis):
    def __init__(
            self,
            host: str | None = '127.0.0.1',
            port: str | None = '6379'
    ):
        super().__init__(
            client="async",
            host=host,
            port=port
        )
        self._pool = AsyncConnectionPool(
            host=self._host,
            port=self._port,
            decode_responses=True
        )
