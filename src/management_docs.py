"""
文档管理模块
对应原 rag_system_code.py 中的：
    DocumentRecord 数据类 / DocumentManifest 清单管理器
    LocalRAGSystem 的文档增/删/改/同步/状态/监控/批量添加等企业级文档管理方法
以 Mixin 形式提供给主系统类使用，方法实现与原文件保持一致。
"""

import os
import json
import hashlib
import threading
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document as LangchainDocument

from config import DOCS_DIR
from process_files import make_chunk_id


# ================== 文档清单管理 ==================
@dataclass
class DocumentRecord:
    """单文档记录"""
    file_path: str
    file_hash: str
    last_modified: float
    chunk_ids: list[str] = field(default_factory=list)
    chunk_count: int = 0
    file_type: str = ""
    indexed_at: str = ""


class DocumentManifest:
    """
    企业级文档清单管理器

    原理：
    - 维护已入库文档的哈希值和元数据映射，实现增量更新
    - 通过文件哈希(MD5) + 修改时间 双重检测文件变更
    - 支持原子性写入（先写临时文件再rename），防止写入中断导致清单损坏
    - 每次变更前自动备份，支持回滚
    """

    def __init__(self, manifest_path: str):
        self._path = manifest_path
        self._records: dict[str, DocumentRecord] = {}
        self._lock = None  # 线程锁，实际使用时由RAGSystem的锁控制

    def load(self) -> bool:
        """加载清单，失败时自动从备份恢复"""
        if not os.path.exists(self._path):
            return False
        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._records = {}
            for file_path, rec in data.get('documents', {}).items():
                self._records[file_path] = DocumentRecord(
                    file_path=rec.get('file_path', file_path),
                    file_hash=rec.get('file_hash', ''),
                    last_modified=rec.get('last_modified', 0),
                    chunk_ids=rec.get('chunk_ids', []),
                    chunk_count=rec.get('chunk_count', 0),
                    file_type=rec.get('file_type', ''),
                    indexed_at=rec.get('indexed_at', ''),
                )
            return True
        except (json.JSONDecodeError, KeyError):
            backup = self._path + '.backup'
            if os.path.exists(backup):
                try:
                    with open(backup, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self._records = {}
                    for file_path, rec in data.get('documents', {}).items():
                        self._records[file_path] = DocumentRecord(**rec)
                    return True
                except Exception:
                    pass
            self._records = {}
            return False

    def save(self) -> None:
        """原子性保存清单（写临时文件 + rename）"""
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        data = {
            'version': '1.0',
            'updated_at': datetime.now().isoformat(),
            'total_documents': len(self._records),
            'total_chunks': sum(r.chunk_count for r in self._records.values()),
            'documents': {
                fp: {
                    'file_path': r.file_path,
                    'file_hash': r.file_hash,
                    'last_modified': r.last_modified,
                    'chunk_ids': r.chunk_ids,
                    'chunk_count': r.chunk_count,
                    'file_type': r.file_type,
                    'indexed_at': r.indexed_at,
                }
                for fp, r in self._records.items()
            }
        }
        tmp_path = self._path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if os.path.exists(self._path):
            try:
                os.replace(self._path, self._path + '.backup')
            except OSError:
                pass

        os.replace(tmp_path, self._path)

    def has_file(self, file_path: str) -> bool:
        return file_path in self._records

    def get_record(self, file_path: str) -> Optional[DocumentRecord]:
        return self._records.get(file_path)

    def add_record(self, record: DocumentRecord) -> None:
        self._records[record.file_path] = record

    def remove_record(self, file_path: str) -> Optional[DocumentRecord]:
        return self._records.pop(file_path, None)

    def get_all_paths(self) -> set[str]:
        return set(self._records.keys())

    def get_total_chunks(self) -> int:
        return sum(r.chunk_count for r in self._records.values())

    def get_total_documents(self) -> int:
        return len(self._records)


class DocumentManagementMixin:
    """
    企业级文档管理能力（供 LocalRAGSystem 多继承组合）

    核心原则：
      1. 基于文件哈希(MD5) + 修改时间 双重变更检测
      2. 使用确定性ID (source_path#chunk_N) 支持精确增删改
      3. 清单持久化，每次变更原子写入
      4. 操作加锁，防止并发冲突
      5. 变更后自动重建BM25和关键词索引
    """

    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        """计算文件MD5哈希（大文件分块读取）"""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()

    def _reload_chunks_from_vectorstore(self) -> None:
        """从向量库全量重载chunks列表（增量操作后保持一致性）"""
        all_docs_data = self.vector_store.get()
        self.chunks = []
        if all_docs_data and 'documents' in all_docs_data:
            for doc_id, text, metadata in zip(
                all_docs_data.get('ids', []),
                all_docs_data.get('documents', []),
                all_docs_data.get('metadatas', []),
            ):
                doc = LangchainDocument(
                    page_content=text,
                    metadata=metadata or {}
                )
                self.chunks.append(doc)

        for i, chunk in enumerate(self.chunks):
            chunk.metadata["chunk_index"] = i

    def _rebuild_indices(self, add_chunks=None, add_start_idx=0, remove_indices=None) -> None:
        """
        重建检索索引。
        - BM25 库不支持增量，始终全量重建
        - 倒排索引优先走增量路径（add_chunks / remove_indices），否则全量重建
        """
        if self.chunks:
            self._build_bm25_index()
            if add_chunks is not None:
                self._add_to_keyword_index(add_chunks, add_start_idx)
            elif remove_indices is not None:
                self._remove_from_keyword_index(remove_indices)
            else:
                self._build_keyword_index()
        else:
            self._bm25_index = None
            self._keyword_index = {}

    def _rebuild_manifest_from_chunks(self) -> None:
        """从现有chunks重建清单（用于首次建库或清单丢失时）"""
        source_groups: dict[str, list] = {}
        for i, chunk in enumerate(self.chunks):
            source = chunk.metadata.get('source', '')
            if source:
                source_groups.setdefault(source, []).append(i)

        for file_path, chunk_indices in source_groups.items():
            file_hash = ''
            last_modified = 0.0
            if os.path.exists(file_path):
                try:
                    file_hash = self._compute_file_hash(file_path)
                    last_modified = os.path.getmtime(file_path)
                except OSError:
                    pass

            chunk_ids = [
                self.chunks[idx].metadata.get("chunk_id", f"{file_path}#chunk_{idx}")
                for idx in chunk_indices
            ]
            self._manifest.add_record(DocumentRecord(
                file_path=file_path,
                file_hash=file_hash,
                last_modified=last_modified,
                chunk_ids=chunk_ids,
                chunk_count=len(chunk_indices),
                file_type=os.path.splitext(file_path)[1].lower(),
                indexed_at=datetime.now().isoformat(),
            ))

        self._manifest.save()
        print(f"清单已重建: {self._manifest.get_total_documents()} 个文件, "
              f"{self._manifest.get_total_chunks()} 个chunk")

    def add_document(self, file_path: str) -> bool:
        """
        企业级文档添加：
        1. 检测文件有效性
        2. 加载文档（支持txt/pdf/docx/doc/md）
        3. 切分为chunks，生成确定性ID
        4. 写入向量库
        5. 更新chunks列表和清单
        6. 重建检索索引
        """
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return False

        abs_path = os.path.abspath(file_path)
        with self._mutation_lock:
            return self._add_document_locked(abs_path)

    def _add_document_locked(self, file_path: str) -> bool:
        """持有锁的文档添加实现"""
        file_hash = self._compute_file_hash(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        if self._manifest.has_file(file_path):
            existing = self._manifest.get_record(file_path)
            if existing and existing.file_hash == file_hash:
                print(f"文档已存在且未变更，跳过: {os.path.basename(file_path)}")
                return True
            else:
                print(f"检测到文档已变更，先移除旧版本: {os.path.basename(file_path)}")
                self._remove_document_locked(file_path)

        try:
            if ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
                raw_docs = loader.load()
            elif ext == '.pdf':
                raw_docs = self._load_pdf_with_fallback(file_path)
            elif ext in ('.docx', '.doc'):
                doc = self._load_docx_with_heading(file_path)
                raw_docs = [doc]
            elif ext in ('.md', '.markdown'):
                raw_docs = self._load_markdown(file_path)
            else:
                print(f"不支持的文件类型: {ext}")
                return False
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            return False

        if not raw_docs:
            print(f"文件无有效内容: {file_path}")
            return False

        chunks = self._split_documents(raw_docs)
        if not chunks:
            print(f"切分后无有效文本块: {file_path}")
            return False

        current_max_idx = len(self.chunks)
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.metadata.get("chunk_id") or make_chunk_id(
                os.path.basename(file_path), (chunk.page_content or "").strip()
            )
            chunk.metadata["chunk_id"] = chunk_id
            chunk_ids.append(chunk_id)
            chunk.metadata["source"] = file_path
            chunk.metadata["chunk_index"] = current_max_idx + i

        try:
            self.vector_store.add_documents(chunks, ids=chunk_ids)
        except Exception as e:
            print(f"写入向量库失败: {e}")
            return False

        self._reload_chunks_from_vectorstore()

        self._manifest.add_record(DocumentRecord(
            file_path=file_path,
            file_hash=file_hash,
            last_modified=os.path.getmtime(file_path),
            chunk_ids=chunk_ids,
            chunk_count=len(chunks),
            file_type=ext,
            indexed_at=datetime.now().isoformat(),
        ))
        self._manifest.save()

        self._rebuild_indices(add_chunks=chunks, add_start_idx=current_max_idx)

        print(f"文档已添加: {os.path.basename(file_path)} ({len(chunks)} chunks)")
        return True

    def remove_document(self, file_path: str) -> bool:
        """
        企业级文档删除：
        1. 从向量库按source过滤删除所有相关chunks
        2. 从清单中移除记录
        3. 重载chunks列表
        4. 重建检索索引
        """
        abs_path = os.path.abspath(file_path)
        with self._mutation_lock:
            return self._remove_document_locked(abs_path)

    def _remove_document_locked(self, file_path: str) -> bool:
        """持有锁的文档删除实现"""
        if not self._manifest.has_file(file_path):
            print(f"文档不在知识库中: {os.path.basename(file_path)}")
            return False

        record = self._manifest.get_record(file_path)
        chunk_count = record.chunk_count if record else 0

        try:
            self.vector_store.delete(where={"source": file_path})
        except Exception as e:
            print(f"从向量库删除失败 {file_path}: {e}")
            return False

        self._manifest.remove_record(file_path)
        self._manifest.save()

        self._reload_chunks_from_vectorstore()
        self._rebuild_indices()  # 全量重建：chunk索引移除后全部偏移，无安全增量路径

        print(f"文档已移除: {os.path.basename(file_path)} ({chunk_count} chunks)")
        return True

    def update_document(self, file_path: str) -> bool:
        """
        企业级文档更新（哈希变更检测 + 先删后加）：
        1. 检查文件是否存在及哈希是否变化
        2. 若已入库且未变化，跳过
        3. 若已入库但已变化，先删除旧版再添加新版
        4. 若未入库，直接添加
        """
        if not os.path.exists(file_path):
            print(f"文件不存在，尝试从知识库中移除: {file_path}")
            return self.remove_document(file_path)

        abs_path = os.path.abspath(file_path)
        with self._mutation_lock:
            if self._manifest.has_file(abs_path):
                current_hash = self._compute_file_hash(abs_path)
                existing = self._manifest.get_record(abs_path)
                if existing and existing.file_hash == current_hash:
                    print(f"文档未变更，跳过: {os.path.basename(abs_path)}")
                    return True
                self._remove_document_locked(abs_path)

            return self._add_document_locked(abs_path)

    def sync_knowledge_base(self) -> dict:
        """
        企业级知识库同步：
        扫描文档目录，与清单比对，批量增/删/改。

        返回统计字典: {added, removed, updated, skipped, errors}
        """
        stats = {'added': 0, 'removed': 0, 'updated': 0, 'skipped': 0, 'errors': 0}
        docs_path = Path(DOCS_DIR)

        if not docs_path.exists():
            print(f"文档目录不存在，创建: {DOCS_DIR}")
            docs_path.mkdir(parents=True)
            return stats

        with self._mutation_lock:
            current_files = set()
            for ext in ('*.txt', '*.pdf', '*.docx', '*.doc', '*.md', '*.markdown'):
                for f in docs_path.glob(ext):
                    current_files.add(str(f.resolve()))

            indexed_files = self._manifest.get_all_paths()

            removed_files = indexed_files - current_files
            for fp in sorted(removed_files):
                try:
                    if self._remove_document_locked(fp):
                        stats['removed'] += 1
                    else:
                        stats['errors'] += 1
                except Exception as e:
                    print(f"删除文档失败 {fp}: {e}")
                    stats['errors'] += 1

            for fp in sorted(current_files):
                try:
                    if self._manifest.has_file(fp):
                        current_hash = self._compute_file_hash(fp)
                        existing = self._manifest.get_record(fp)
                        if existing and existing.file_hash == current_hash:
                            stats['skipped'] += 1
                            continue
                        if self._remove_document_locked(fp):
                            if self._add_document_locked(fp):
                                stats['updated'] += 1
                            else:
                                stats['errors'] += 1
                        else:
                            stats['errors'] += 1
                    else:
                        if self._add_document_locked(fp):
                            stats['added'] += 1
                        else:
                            stats['errors'] += 1
                except Exception as e:
                    print(f"同步文档失败 {fp}: {e}")
                    stats['errors'] += 1

        print(f"\n知识库同步完成: "
              f"新增{stats['added']}, 移除{stats['removed']}, "
              f"更新{stats['updated']}, 跳过{stats['skipped']}, "
              f"失败{stats['errors']}")

        self._reload_chunks_from_vectorstore()
        self._rebuild_indices()  # 批量同步后全量重建，确保一致性

        return stats

    def get_document_status(self) -> dict:
        """获取知识库文档状态报告"""
        manifest_info = {
            'total_documents': self._manifest.get_total_documents(),
            'total_chunks': self._manifest.get_total_chunks(),
            'vector_count': self.vector_store._collection.count() if self.vector_store else 0,
        }

        documents = []
        for fp in sorted(self._manifest.get_all_paths()):
            rec = self._manifest.get_record(fp)
            exists = os.path.exists(fp)
            current_hash = ''
            if exists:
                try:
                    current_hash = self._compute_file_hash(fp)
                except OSError:
                    pass
            documents.append({
                'file': os.path.basename(fp),
                'path': fp,
                'type': rec.file_type if rec else '',
                'chunks': rec.chunk_count if rec else 0,
                'indexed_at': rec.indexed_at if rec else '',
                'on_disk': exists,
                'hash_match': rec.file_hash == current_hash if rec and exists else None,
            })

        return {
            'manifest': manifest_info,
            'documents': documents,
        }

    def start_file_watcher(self, poll_interval: float = 5.0) -> threading.Thread:
        """
        启动后台文件监控线程（基于轮询）。

        原理：定时扫描文档目录，检测文件增删改并自动同步到向量库。
        适用于生产环境中文档持续更新的场景。

        :param poll_interval: 轮询间隔（秒）
        :return: 监控线程对象，可调用 .join() 阻塞或 .stop() 停止
        """
        stop_event = threading.Event()

        def _watch_loop():
            print(f"文件监控已启动 (轮询间隔: {poll_interval}s, 目录: {DOCS_DIR})")
            last_state = {}
            docs_path = Path(DOCS_DIR)

            while not stop_event.is_set():
                try:
                    if not docs_path.exists():
                        stop_event.wait(poll_interval)
                        continue

                    current_state = {}
                    for ext in ('*.txt', '*.pdf', '*.docx', '*.doc', '*.md', '*.markdown'):
                        for f in docs_path.glob(ext):
                            fp = str(f.resolve())
                            try:
                                current_state[fp] = {
                                    'hash': self._compute_file_hash(fp),
                                    'mtime': os.path.getmtime(fp),
                                }
                            except OSError:
                                continue

                    if last_state:
                        current_set = set(current_state.keys())
                        last_set = set(last_state.keys())

                        added = current_set - last_set
                        removed = last_set - current_set
                        modified = {
                            fp for fp in (current_set & last_set)
                            if current_state[fp]['hash'] != last_state[fp].get('hash', '')
                        }

                        for fp in sorted(added):
                            print(f"[监控] 检测到新增文件: {os.path.basename(fp)}")
                            self.add_document(fp)

                        for fp in sorted(removed):
                            print(f"[监控] 检测到删除文件: {os.path.basename(fp)}")
                            self.remove_document(fp)

                        for fp in sorted(modified):
                            print(f"[监控] 检测到修改文件: {os.path.basename(fp)}")
                            self.update_document(fp)

                    last_state = current_state
                except Exception as e:
                    print(f"[监控] 扫描异常: {e}")

                stop_event.wait(poll_interval)

            print("文件监控已停止")

        watcher_thread = threading.Thread(target=_watch_loop, daemon=True, name="FileWatcher")
        watcher_thread._stop_event = stop_event
        watcher_thread.start()
        return watcher_thread

    def add_documents_batch(self, file_paths: str) -> dict:
        """
        批量添加文档（企业级批量处理）。

        优化：多个文档共享一次索引重建，而非每个文档重建一次。

        :param file_paths: 文档路径列表
        :return: 统计字典
        """
        stats = {'added': 0, 'skipped': 0, 'errors': 0}
        print(file_paths)
        with self._mutation_lock:
            for fp in os.listdir(os.path.join(file_paths)):
                print(fp)
                try:
                    result = self._add_document_locked(os.path.join(file_paths, fp))
                    if result:
                        stats['added'] += 1
                    else:
                        stats['errors'] += 1
                except Exception as e:
                    print(f"批量添加失败 {fp}: {e}")
                    stats['errors'] += 1

        self._reload_chunks_from_vectorstore()
        self._rebuild_indices()  # 批量操作后统一全量重建，避免多次重复构建

        print(f"\n批量添加完成: 成功{stats['added']}, 跳过{stats['skipped']}, 失败{stats['errors']}")
        return stats
