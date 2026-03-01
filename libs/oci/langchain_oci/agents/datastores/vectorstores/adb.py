# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Oracle Autonomous Database vector store datastore."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_oci.agents.datastores.vectorstores.base import VectorDataStore


@dataclass
class ADB(VectorDataStore):
    """Oracle Autonomous Database vector datastore.

    Example:
        >>> from langchain_oci.agents import ADB, create_datastore_tools
        >>>
        >>> store = ADB(
        ...     dsn="mydb_low",
        ...     user="ADMIN",
        ...     password="...",
        ...     wallet_location="~/.oracle-wallet",
        ...     hint="sales data, revenue, customers",
        ... )
        >>>
        >>> tools = create_datastore_tools(
        ...     stores={"sales": store},
        ...     compartment_id="ocid1.compartment...",
        ... )
    """

    dsn: str
    user: str
    password: str
    wallet_location: Optional[str] = None
    wallet_password: Optional[str] = None
    table_name: str = "VECTOR_DOCUMENTS"
    hint: str = ""

    _connection: Any = field(default=None, repr=False)
    _embedding_model: Any = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "adb"

    def connect(self, embedding_model: Any) -> None:
        try:
            import oracledb
        except ImportError as e:
            raise ImportError("oracledb required: pip install oracledb") from e

        config_dir = None
        if self.wallet_location:
            config_dir = os.path.expanduser(self.wallet_location)

        self._connection = oracledb.connect(
            user=self.user,
            password=self.password,
            dsn=self.dsn,
            config_dir=config_dir,
            wallet_location=config_dir,
            wallet_password=self.wallet_password or self.password,
        )
        self._embedding_model = embedding_model

    def _read_clob(self, value: Any) -> str:
        if hasattr(value, "read"):
            return value.read()
        return str(value) if value else ""

    def search(self, query: str, embedding: list[float], top_k: int) -> list[dict]:
        cursor = self._connection.cursor()
        sql = f"""
            SELECT id, title, content, source,
                   VECTOR_DISTANCE(embedding, :query_vec, COSINE) as distance
            FROM {self.table_name}
            ORDER BY VECTOR_DISTANCE(embedding, :query_vec, COSINE)
            FETCH FIRST :top_k ROWS ONLY
        """
        cursor.execute(sql, {"query_vec": json.dumps(embedding), "top_k": top_k})
        results = []
        for row in cursor:
            results.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "content": self._read_clob(row[2])[:1000],
                    "source": row[3],
                    "score": 1 - row[4],
                }
            )
        cursor.close()
        return results

    def keyword_search(self, query: str, top_k: int) -> list[dict]:
        cursor = self._connection.cursor()
        sql = f"""
            SELECT id, title, content, source
            FROM {self.table_name}
            WHERE UPPER(content) LIKE UPPER(:pattern)
               OR UPPER(title) LIKE UPPER(:pattern)
            FETCH FIRST :top_k ROWS ONLY
        """
        cursor.execute(sql, {"pattern": f"%{query}%", "top_k": top_k})
        results = []
        for row in cursor:
            results.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "content": self._read_clob(row[2])[:1000],
                    "source": row[3],
                }
            )
        cursor.close()
        return results

    def get(self, document_id: str | int) -> Optional[dict]:
        cursor = self._connection.cursor()
        sql = f"""
            SELECT id, title, content, source, created_at
            FROM {self.table_name}
            WHERE id = :id
        """
        cursor.execute(sql, {"id": int(document_id)})
        row = cursor.fetchone()
        cursor.close()
        if not row:
            return None
        return {
            "id": row[0],
            "title": row[1],
            "content": self._read_clob(row[2]),
            "source": row[3],
            "created_at": str(row[4]) if row[4] else None,
        }

    def insert(
        self, title: str, content: str, source: str, embedding: list[float]
    ) -> str:
        cursor = self._connection.cursor()
        sql = f"""
            INSERT INTO {self.table_name} (title, content, source, embedding)
            VALUES (:title, :content, :source, :embedding)
            RETURNING id INTO :out_id
        """
        out_id = cursor.var(int)
        cursor.execute(
            sql,
            {
                "title": title,
                "content": content,
                "source": source,
                "embedding": json.dumps(embedding),
                "out_id": out_id,
            },
        )
        self._connection.commit()
        doc_id = out_id.getvalue()[0]
        cursor.close()
        return str(doc_id)

    def bulk_insert(self, documents: list[dict], embeddings: list[list[float]]) -> int:
        cursor = self._connection.cursor()
        sql = f"""
            INSERT INTO {self.table_name} (title, content, source, embedding)
            VALUES (:title, :content, :source, :embedding)
        """
        batch_data = [
            {
                "title": doc.get("title", "Untitled"),
                "content": doc.get("content", ""),
                "source": doc.get("source", "bulk_insert"),
                "embedding": json.dumps(embedding),
            }
            for doc, embedding in zip(documents, embeddings)
        ]
        cursor.executemany(sql, batch_data)
        self._connection.commit()
        count = cursor.rowcount
        cursor.close()
        return count

    def update(
        self,
        document_id: str | int,
        title: Optional[str],
        content: Optional[str],
        source: Optional[str],
        embedding: Optional[list[float]],
    ) -> bool:
        cursor = self._connection.cursor()
        set_clauses = []
        params: dict[str, Any] = {"id": int(document_id)}
        if title is not None:
            set_clauses.append("title = :title")
            params["title"] = title
        if content is not None:
            set_clauses.append("content = :content")
            params["content"] = content
        if source is not None:
            set_clauses.append("source = :source")
            params["source"] = source
        if embedding is not None:
            set_clauses.append("embedding = :embedding")
            params["embedding"] = json.dumps(embedding)
        if not set_clauses:
            return False
        sql = f"UPDATE {self.table_name} SET {', '.join(set_clauses)} WHERE id = :id"
        cursor.execute(sql, params)
        self._connection.commit()
        updated = cursor.rowcount > 0
        cursor.close()
        return updated

    def delete(self, document_id: str | int) -> bool:
        cursor = self._connection.cursor()
        cursor.execute(
            f"DELETE FROM {self.table_name} WHERE id = :id",
            {"id": int(document_id)},
        )
        self._connection.commit()
        deleted = cursor.rowcount > 0
        cursor.close()
        return deleted

    def stats(self) -> dict:
        cursor = self._connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        count = cursor.fetchone()[0]
        cursor.execute(f"""
            SELECT source, COUNT(*) as cnt
            FROM {self.table_name}
            GROUP BY source
            ORDER BY cnt DESC
            FETCH FIRST 10 ROWS ONLY
        """)
        sources = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()
        return {
            "store": self.name,
            "table": self.table_name,
            "document_count": count,
            "sources": sources,
        }
