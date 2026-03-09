import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any

import json5

from verl.tools.schemas import ToolResponse
from verl.tools.search_tool import SearchTool
from verl.utils.rollout_trace import rollout_trace_op

from .search_r1_like_utils import perform_single_search_batch

logger = logging.getLogger(__name__)


class SafeDatabaseConnection:
    """Thread-safe SQLite connection manager using WAL mode.

    Relies on SQLite's built-in WAL concurrency (multiple readers, single
    writer) and busy_timeout instead of external file locks. Each thread
    gets its own persistent connection via threading.local().
    """

    def __init__(self, db_path):
        self.db_path = db_path
        self._local = threading.local()

        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self._initialize_database()

    def _initialize_database(self):
        """Initialize or recover the database (called once at startup)."""
        conn = sqlite3.connect(self.db_path, timeout=60)
        try:
            conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        except sqlite3.DatabaseError:
            conn.close()
            self._remove_db_files()
            conn = sqlite3.connect(self.db_path, timeout=60)

        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA busy_timeout=60000')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS search_cache (
                cache_key TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                results TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_at
            ON search_cache(created_at)
        ''')
        conn.commit()
        conn.close()
        logger.info(f"[SearchCache] Database ready: {self.db_path}")

    def _remove_db_files(self):
        """Backup and remove corrupted database and WAL/SHM files."""
        backup_ts = int(time.time())
        for ext in ('', '-wal', '-shm'):
            path = f"{self.db_path}{ext}"
            if os.path.exists(path):
                backup = f"{self.db_path}.corrupted.{backup_ts}{ext}"
                os.rename(path, backup)
                logger.warning(f"[SearchCache] Moved corrupted file {path} -> {backup}")

    def _get_thread_conn(self) -> sqlite3.Connection:
        """Get or create a persistent connection for the current thread."""
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            try:
                conn.execute('SELECT 1')
                return conn
            except sqlite3.Error:
                try:
                    conn.close()
                except Exception:
                    pass
                self._local.conn = None

        conn = sqlite3.connect(
            self.db_path,
            timeout=60,
            isolation_level='DEFERRED',
            check_same_thread=False,
        )
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA busy_timeout=60000')
        conn.execute('PRAGMA cache_size=-64000')
        self._local.conn = conn
        return conn

    @contextmanager
    def get_connection(self):
        """Yield a thread-local database connection."""
        conn = self._get_thread_conn()
        try:
            yield conn
        except sqlite3.DatabaseError:
            self._local.conn = None
            try:
                conn.close()
            except Exception:
                pass
            raise


class CustomSearchTool(SearchTool):
    """Search tool with SQLite-backed result cache."""

    MAX_CACHE_TOPK = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.db_file_path = os.getenv(
            'DB_FILE_PATH',
            os.getenv('SEARCH_CACHE_DB_PATH', './search_cache.db')
        )
        self.safe_db = SafeDatabaseConnection(self.db_file_path)

    def _get_cache_key(self, query: str) -> str:
        normalized_query = query.strip()
        return hashlib.md5(normalized_query.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> list | None:
        try:
            with self.safe_db.get_connection() as conn:
                cursor = conn.execute(
                    'SELECT results FROM search_cache WHERE cache_key = ?',
                    (cache_key,)
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None
        except Exception as e:
            logger.warning(f"[SearchCache] Cache read failed: {e}")
            return None

    def _save_to_cache(self, cache_key: str, query: str, results: list):
        try:
            with self.safe_db.get_connection() as conn:
                conn.execute(
                    'INSERT OR REPLACE INTO search_cache (cache_key, query, results) VALUES (?, ?, ?)',
                    (cache_key, query.strip(), json.dumps(results, ensure_ascii=False))
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"[SearchCache] Cache write failed: {e}")

    def _parse_search_results(self, result_text: str) -> list[dict]:
        """Parse search API response text into a list of result dicts."""
        try:
            results = json5.loads(result_text)
            if isinstance(results, list):
                return results
            if isinstance(results, dict) and 'results' in results:
                return results['results']
            if isinstance(results, dict):
                return [results]
        except (json5.JSON5DecodeError, ValueError):
            pass

        lines = result_text.strip().split('\n')
        return [{'text': line} for line in lines if line.strip()]

    def _format_search_results(self, results: list[dict]) -> str:
        return json.dumps(results, ensure_ascii=False)

    def execute_search(self, instance_id: str, query: str, retrieval_service_url: str,
                       topk: int, timeout: int) -> tuple[str, dict]:
        """Execute search with caching.

        Returns:
            Tuple of (result_text, metadata)
        """
        cache_key = self._get_cache_key(query)

        cached_results = self._get_from_cache(cache_key)
        if cached_results is not None:
            sliced = cached_results[:topk]
            return self._format_search_results(sliced), {
                'from_cache': True,
                'requested_topk': topk,
                'cached_topk': len(cached_results),
                'total_results': len(sliced),
                'status': 'cache_hit',
                'query_count': 0,
                'api_request_error': None,
            }

        search_topk = max(topk, self.MAX_CACHE_TOPK)
        result_text, metadata = perform_single_search_batch(
            retrieval_service_url=retrieval_service_url,
            query=query,
            topk=search_topk,
            concurrent_semaphore=None,
            timeout=timeout,
        )

        search_results = self._parse_search_results(result_text)

        if metadata.get("status") == "success" and search_results:
            self._save_to_cache(cache_key, query, search_results)

        if topk < search_topk and search_results:
            sliced = search_results[:topk]
            result_text = self._format_search_results(sliced)
            metadata['requested_topk'] = topk
            metadata['total_results'] = len(sliced)

        metadata['from_cache'] = False
        metadata['cached_topk'] = search_topk
        return result_text, metadata

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query_from_params = parameters.get("query")

        if not query_from_params:
            error_msg = "Error: 'query' is missing or empty."
            logger.error(f"[SearchTool] {error_msg} Parameters: {parameters}")
            return (
                ToolResponse(text=json.dumps({"error": error_msg})),
                0.0,
                {"status": "error", "error": "missing_query"},
            )

        try:
            result_text, metadata = await self.execution_pool.execute.remote(
                self.execute_search,
                instance_id,
                query_from_params,
                self.retrieval_service_url,
                self.topk,
                self.timeout,
            )

            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            metrics = {
                "query_count": metadata.get("query_count", 0),
                "status": metadata.get("status", "unknown"),
                "total_results": metadata.get("total_results", 0),
                "api_request_error": metadata.get("api_request_error"),
                "from_cache": metadata.get("from_cache", False),
                "requested_topk": metadata.get("requested_topk", self.topk),
                "cached_topk": metadata.get("cached_topk", self.topk),
            }
            return ToolResponse(text=result_text), 0.0, metrics

        except Exception as e:
            error_msg = f"Search execution failed: {e}"
            logger.error(f"[SearchTool] {error_msg}", exc_info=True)
            return (
                ToolResponse(text=json.dumps({"error": error_msg})),
                0.0,
                {"status": "error", "error": str(e), "from_cache": False},
            )
