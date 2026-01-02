import fcntl
import hashlib
import logging
import os
import sqlite3
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
    def __init__(self, db_path):
        self.db_path = db_path
        self.lock_file_path = f"{db_path}.lock"
        self._initialized = False
        
        # 确保数据库目录存在
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        self._initialize_database_once()
    
    def _initialize_database_once(self):
        """一次性初始化数据库（仅首次，带全局锁）"""
        lock_file = None
        try:
            # 1. 获取全局初始化锁
            lock_file = open(self.lock_file_path, 'w')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            
            # 2. 检查数据库是否已存在且有效
            db_exists = os.path.exists(self.db_path)
            
            # 3. 创建或验证数据库
            conn = sqlite3.connect(
                self.db_path,
                timeout=30,
                isolation_level='DEFERRED'
            )
            
            # 4. ⭐ 只在数据库不存在或未初始化时设置 WAL 模式
            if not db_exists:
                logger.info(f"[SearchCache] Initializing new database: {self.db_path}")
                
                # 设置 WAL 模式（仅一次）
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('PRAGMA synchronous=NORMAL')
                
                # 创建表
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
                logger.info("[SearchCache] Database initialized successfully")
            else:
                # 验证现有数据库
                cursor = conn.execute("PRAGMA integrity_check")
                result = cursor.fetchone()[0]
                if result != 'ok':
                    raise sqlite3.DatabaseError(f"Database integrity check failed: {result}")
                
                # 确保 WAL 模式已启用（检查，不强制设置）
                cursor = conn.execute("PRAGMA journal_mode")
                current_mode = cursor.fetchone()[0]
                if current_mode.upper() != 'WAL':
                    logger.warning(f"[SearchCache] Database is not in WAL mode ({current_mode}), converting...")
                    conn.execute('PRAGMA journal_mode=WAL')
            
            conn.close()
            self._initialized = True
            
        except Exception as e:
            logger.error(f"[SearchCache] Database initialization failed: {e}")
            self._initialized = False
            raise
        finally:
            if lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
    
    @contextmanager
    def get_connection(self, timeout=30, max_retries=3):
        """获取数据库连接（带重试机制）"""
        lock_file = None
        conn = None
        
        for attempt in range(max_retries):
            try:
                # 1. 获取文件锁
                lock_file = open(self.lock_file_path, 'w')
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                
                # 2. 连接数据库
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=timeout,
                    isolation_level='DEFERRED',
                    check_same_thread=False
                )
                
                # 3. ⭐ 不再重复设置 PRAGMA，只设置会话级参数
                conn.execute('PRAGMA busy_timeout=30000')
                conn.execute('PRAGMA temp_store=MEMORY')
                conn.execute('PRAGMA cache_size=-64000')
                
                # 4. 测试连接
                conn.execute('SELECT 1').fetchone()
                
                yield conn
                break
                
            except sqlite3.DatabaseError as e:
                logger.error(f"[SearchCache] Database error (attempt {attempt + 1}/{max_retries}): {e}")
                
                if conn:
                    try:
                        conn.close()
                    except:
                        pass
                    conn = None
                
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # 指数退避
                    continue
                else:
                    # 最后一次尝试失败，检查是否需要恢复
                    self._handle_corrupted_database()
                    raise
                    
            except Exception as e:
                logger.error(f"[SearchCache] Unexpected error: {e}")
                raise
                
            finally:
                # 清理资源
                if conn:
                    try:
                        conn.close()
                    except Exception as e:
                        logger.warning(f"[SearchCache] Error closing connection: {e}")
                
                if lock_file:
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        lock_file.close()
                    except Exception as e:
                        logger.warning(f"[SearchCache] Error releasing lock: {e}")
    
    def _handle_corrupted_database(self):
        """处理损坏的数据库"""
        try:
            if os.path.exists(self.db_path):
                backup_path = f"{self.db_path}.corrupted.{int(time.time())}"
                
                # 也备份 WAL 文件
                for ext in ['', '-wal', '-shm']:
                    src = f"{self.db_path}{ext}"
                    if os.path.exists(src):
                        dst = f"{backup_path}{ext}"
                        os.rename(src, dst)
                
                logger.warning(f"[SearchCache] Moved corrupted database to {backup_path}")
                
                # 重新初始化
                self._initialized = False
                self._initialize_database_once()
                
        except Exception as e:
            logger.error(f"[SearchCache] Failed to handle corrupted database: {e}")


class CustomSearchTool(SearchTool):
    """带缓存的搜索工具（进程安全）"""
    MAX_CACHE_TOPK = 10  # 缓存时使用的最大topk值

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.db_file_path = os.getenv(
            'DB_FILE_PATH', 
            os.getenv('SEARCH_CACHE_DB_PATH', './search_cache.db')
        )
        
        # ⭐ 初始化安全数据库连接（会自动完成数据库初始化）
        self.safe_db = SafeDatabaseConnection(self.db_file_path)
        self._cache_enabled = True

    def _get_cache_key(self, query: str) -> str:
        """生成缓存键（基于query.strip()）"""
        normalized_query = query.strip()
        return hashlib.md5(normalized_query.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> list | None:
        """从缓存获取结果（进程安全）"""
        if not self._cache_enabled:
            return None
            
        try:
            with self.safe_db.get_connection() as conn:
                cursor = conn.execute(
                    'SELECT results FROM search_cache WHERE cache_key = ?',
                    (cache_key,)
                )
                row = cursor.fetchone()
                
                if row:
                    results = json5.loads(row[0])
                    logger.debug(f"[SearchCache] Cache hit for key: {cache_key}")
                    return results
                    
                return None
                
        except sqlite3.DatabaseError as e:
            logger.error(f"[SearchCache] Database error reading cache: {e}")
            self._cache_enabled = False
            return None
            
        except Exception as e:
            logger.error(f"[SearchCache] Failed to read from cache: {e}")
            return None

    def _save_to_cache(self, cache_key: str, query: str, results: list):
        """保存到缓存（进程安全）"""
        if not self._cache_enabled:
            return
            
        try:
            with self.safe_db.get_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO search_cache 
                    (cache_key, query, results)
                    VALUES (?, ?, ?)
                ''', (
                    cache_key, 
                    query.strip(), 
                    json5.dumps(results, ensure_ascii=False)
                ))
                conn.commit()
                logger.debug(f"[SearchCache] Saved cache for query: {query.strip()[:50]}...")
                
        except sqlite3.DatabaseError as e:
            logger.error(f"[SearchCache] Database error saving cache: {e}")
            self._cache_enabled = False
            
        except Exception as e:
            logger.error(f"[SearchCache] Failed to save to cache: {e}")

    def _parse_search_results(self, result_text: str) -> list[dict]:
        """解析搜索结果文本为列表"""
        try:
            results = json5.loads(result_text)
            
            if isinstance(results, list):
                return results
            elif isinstance(results, dict):
                if 'results' in results:
                    return results['results']
                return [results]
                
        except (json5.JSON5DecodeError, ValueError) as e:
            logger.warning(f"[SearchCache] Failed to parse JSON5: {e}")
        
        # 降级到文本解析
        lines = result_text.strip().split('\n')
        return [{'text': line} for line in lines if line.strip()]

    def _format_search_results(self, results: list[dict]) -> str:
        """将搜索结果列表格式化为文本"""
        try:
            return json5.dumps(results, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[SearchCache] Failed to serialize results: {e}")
            # 降级到简单文本格式
            return '\n'.join([r.get('text', str(r)) for r in results])

    def _slice_results(self, results: list[dict], requested_topk: int) -> list[dict]:
        """从结果列表中切片出所需数量的结果"""
        return results[:requested_topk]

    def execute_search(self, instance_id: str, query: str, retrieval_service_url: str, 
                      topk: int, timeout: int) -> tuple[str, dict]:
        """Execute search operation using retrieval service with caching.

        Args:
            instance_id: Tool instance ID
            query: search query
            retrieval_service_url: URL of the retrieval service
            topk: Number of top results to return
            timeout: Request timeout in seconds

        Returns:
            Tuple of (result_text, metadata)
        """
        # 生成缓存键
        cache_key = self._get_cache_key(query)
        
        # 尝试从缓存获取
        cached_results = self._get_from_cache(cache_key)
        if cached_results is not None:
            sliced_results = self._slice_results(cached_results, topk)
            result_text = self._format_search_results(sliced_results)
            
            metadata = {
                'from_cache': True,
                'requested_topk': topk,
                'cached_topk': len(cached_results),
                'total_results': len(sliced_results),
                'status': 'cache_hit',
                'query_count': 0,
                'api_request_error': None
            }
            
            print("[SearchCache] Cache hit.")
            return result_text, metadata
        
        # 缓存未命中，执行实际搜索
        search_topk = max(topk, self.MAX_CACHE_TOPK)
        
        result_text, metadata = perform_single_search_batch(
            retrieval_service_url=retrieval_service_url,
            query=query,
            topk=search_topk,
            concurrent_semaphore=None,
            timeout=timeout,
        )
        
        # 解析并缓存搜索结果
        search_results = self._parse_search_results(result_text)
        
        if metadata.get("status") == "success" and search_results:
            self._save_to_cache(cache_key, query, search_results)
        
        # 如果需要，切片结果
        if topk < search_topk and search_results:
            sliced_results = self._slice_results(search_results, topk)
            result_text = self._format_search_results(sliced_results)
            metadata['requested_topk'] = topk
            metadata['total_results'] = len(sliced_results)
        
        # 更新metadata
        metadata['from_cache'] = False
        metadata['cached_topk'] = search_topk

        logger.debug(f"[SearchCache] Search completed for {instance_id}: {len(search_results)} results")
        return result_text, metadata
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query and optional timeout

        Returns:
            Tuple of (tool_response, tool_reward_score, tool_metrics)
        """
        query_from_params = parameters.get("query")

        if not query_from_params:
            error_msg = "Error: 'query' is missing or empty."
            logger.error(f"[SearchTool] {error_msg} Parameters: {parameters}")
            return (
                ToolResponse(text=json5.dumps({"error": error_msg})), 
                0.0, 
                {"status": "error", "error": "missing_query"}
            )

        try:
            # Execute search using Ray execution pool
            result_text, metadata = await self.execution_pool.execute.remote(
                self.execute_search, 
                instance_id, 
                query_from_params, 
                self.retrieval_service_url, 
                self.topk, 
                self.timeout
            )

            # Store results in instance dictionary
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            # Convert metadata to metrics
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
            error_msg = f"Search execution failed: {str(e)}"
            logger.error(f"[SearchTool] {error_msg}", exc_info=True)
            
            return (
                ToolResponse(text=json5.dumps({"error": error_msg})),
                0.0,
                {
                    "status": "error",
                    "error": str(e),
                    "from_cache": False
                }
            )