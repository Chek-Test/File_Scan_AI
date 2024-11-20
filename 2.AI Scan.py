import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QLineEdit, QPushButton, QTextEdit, QWidget
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import numpy as np
import openai
import time

# 設定 OpenAI API 金鑰
openai.api_key = "你的 API"

# 模型初始化
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 模擬檔案資料庫（實際可用 SQLite 或其他）
def setup_sample_database():
    conn = sqlite3.connect("file_index.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            name TEXT,
            path TEXT,
            extension TEXT,  -- 使用 'extension' 欄位
            mime_type TEXT,  -- 使用 'mime_type' 欄位
            embedding BLOB
        )
    """)
    # 插入範例資料
    sample_data = []
    cursor.executemany("""
        INSERT OR REPLACE INTO files (name, path, extension, mime_type, embedding)
        VALUES (?, ?, ?, ?, ?)
    """, sample_data)
    conn.commit()
    conn.close()

# 搜尋程式邏輯
class FileSearchAI:
    def __init__(self):
        self.conn = sqlite3.connect("file_index.db")
    
    # # 原始版本LLM調用
    # def search_by_nlp(self, user_input):
    #     prompt = f"用這段話提取搜尋條件：'{user_input}'"
    #     response = openai.completions.create(
    #         model="gpt-3.5-turbo",
    #         prompt=prompt,
    #         max_tokens=100,  # 你可以根據需求調整這個數值
    #         temperature=0.7   # 控制生成文本的隨機性
    #     )
    #     return response['choices'][0]['text'].strip()
    
    # 總結版本 流量較低
    def search_by_nlp(self, user_input):
        prompt = f"用這段話提取搜尋條件：'{user_input}'"
        response = openai.completions.create(
            model="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        )
        time.sleep(1)  # 添加延遲，減少請求頻率
        return response['choices'][0]['text'].strip()
    
    def search_by_embedding(self, user_input):
        user_embedding = np.array(embedding_model.encode(user_input))
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, path, extension, embedding FROM files")  # 修改 'type' 為 'extension'
        results = []
        for name, path, extension, embedding_blob in cursor.fetchall():
            file_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            similarity = cosine_similarity([user_embedding], [file_embedding])[0][0]
            results.append((name, path, extension, similarity))  # 返回 extension 取代 type
        return sorted(results, key=lambda x: x[3], reverse=True)

# GUI 設計
class FileSearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 文件搜尋助手")
        self.setGeometry(100, 100, 600, 400)

        # 搜尋層
        self.search_ai = FileSearchAI()

        # 設計介面
        central_widget = QWidget()
        layout = QVBoxLayout()

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("輸入搜尋關鍵字或描述...")
        layout.addWidget(self.input_box)

        self.search_button = QPushButton("搜尋")
        layout.addWidget(self.search_button)

        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        layout.addWidget(self.result_area)

        self.search_button.clicked.connect(self.perform_search)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
    
    def perform_search(self):
        user_input = self.input_box.text()
        if not user_input.strip():
            self.result_area.setText("請輸入搜尋內容！")
            return
        
        # 嘗試使用 NLP 處理用戶的查詢
        search_query = self.search_ai.search_by_nlp(user_input)  # 調用 NLP 查詢處理
        if not search_query.strip():
            self.result_area.setText("無法解析搜尋條件，請再試一次！")
            return

        # 根據 NLP 提取的條件來進行嵌入搜尋
        results = self.search_ai.search_by_embedding(search_query)  # 使用 NLP 輸出的條件進行嵌入搜尋

        # 結果展示
        if results:
            result_text = "搜尋結果：\n\n"
            for name, path, file_type, similarity in results:
                result_text += f"檔案名稱: {name}\n路徑: {path}\n類型: {file_type}\n相似度: {similarity:.2f}\n\n"
        else:
            result_text = "未找到相關檔案。"

        self.result_area.setText(result_text)


# 主程式入口
if __name__ == "__main__":
    setup_sample_database()  # 初始化範例資料庫
    
    app = QApplication(sys.argv)
    main_window = FileSearchApp()
    main_window.show()
    sys.exit(app.exec())
