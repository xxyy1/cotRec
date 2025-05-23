import atexit
from pathlib import Path
from redaccel.io.cache import kvcache_registry, KVCache


@kvcache_registry.register()
class SQLiteCache(KVCache):
    def __init__(self, cache_file: str):
        from sqlitedict import SqliteDict

        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        self.cache = SqliteDict(cache_file, autocommit=True)
        atexit.register(self.cache.close)

    def set(self, key: str, value: bytes):
        self.cache[key] = value

    def get(self, key: str) -> bytes:
        if key not in self.cache:
            raise KeyError
        return self.cache[key]

    @property
    def group(self) -> str:
        return "default"
