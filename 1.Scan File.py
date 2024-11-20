import os
from pathlib import Path
import sqlite3
import mimetypes
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 初始化語意向量模型
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------------( 掃描資料夾結構 )---------------------------------

def get_detailed_file_info(file_path):
    file_path = Path(file_path)
    try:
        stats = file_path.stat()
        owner = os.getlogin()  # 當前執行程序的使用者
        is_hidden = file_path.name.startswith(".")
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return {
            "name": file_path.name,
            "path": str(file_path),
            "size": stats.st_size,  # 檔案大小 (bytes)
            "modified_time": time.ctime(stats.st_mtime),  # 修改時間
            "created_time": time.ctime(stats.st_ctime),  # 建立時間
            "extension": file_path.suffix.lower(),  # 副檔名
            "owner": owner,  # 檔案擁有者
            "is_hidden": is_hidden,  # 是否為隱藏檔案
            "mime_type": mime_type or "unknown"  # MIME 類型
        }
    except Exception as e:
        print(f"無法取得檔案 {file_path} 的詳細資訊: {e}")
        return None

def scan_directory(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            file_info = get_detailed_file_info(file_path)
            if file_info:  # 排除無法處理的檔案
                file_list.append(file_info)
    return file_list

# 測試掃描功能
directory_to_scan = "C:/Users/User/Downloads"
scanned_files = scan_directory(directory_to_scan)
print(f"找到 {len(scanned_files)} 個檔案")

# ---------------------------------( 建立與更新資料庫 )---------------------------------

def create_database(db_path="file_index.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 刪除舊的表格，如果存在
    cursor.execute("DROP TABLE IF EXISTS files")
    
    # 創建新表格，新增 'embedding' 欄位來儲存語意向量
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            path TEXT NOT NULL UNIQUE,
            size INTEGER,
            modified_time TEXT,
            created_time TEXT,
            extension TEXT,
            owner TEXT,
            is_hidden INTEGER,
            mime_type TEXT,
            embedding BLOB  -- 用來存儲語意向量
        )
    ''')
    conn.commit()
    conn.close()

def insert_files_to_db(file_list, db_path="file_index.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for file_info in file_list:
        # 使用檔案名稱生成語意向量（這裡可以根據需求調整，這裡使用檔案名稱）
        embedding = embedding_model.encode(file_info["name"])  # 生成語意向量
        
        cursor.execute('''
            INSERT OR REPLACE INTO files (
                name, path, size, modified_time, created_time,
                extension, owner, is_hidden, mime_type, embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            file_info["name"], file_info["path"], file_info["size"],
            file_info["modified_time"], file_info["created_time"],
            file_info["extension"], file_info["owner"],
            int(file_info["is_hidden"]), file_info["mime_type"],
            embedding.tobytes()  # 將語意向量轉換為二進位制儲存
        ))
    conn.commit()
    conn.close()

# 創建資料庫並插入掃描的檔案
create_database()
insert_files_to_db(scanned_files)

# ---------------------------------( 搜尋檔案索引 )---------------------------------

def search_files_by_embedding(user_input, db_path="file_index.db"):
    # 生成使用者輸入的語意向量
    user_embedding = np.array(embedding_model.encode(user_input))
    
    # 從資料庫中查詢所有檔案，並計算與輸入的相似度
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, path, embedding FROM files")
    
    results = []
    for id, name, path, embedding_blob in cursor.fetchall():
        # 從資料庫取出並反序列化語意向量
        file_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        # 計算相似度
        similarity = cosine_similarity([user_embedding], [file_embedding])[0][0]
        results.append((name, path, similarity))
    
    conn.close()
    
    # 根據相似度排序並返回
    return sorted(results, key=lambda x: x[2], reverse=True)

# 測試搜尋功能
search_results = search_files_by_embedding("report")
for result in search_results:
    print(f"檔案名稱: {result[0]}, 路徑: {result[1]}, 相似度: {result[2]:.2f}")
