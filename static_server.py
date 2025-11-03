from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

# 创建FastAPI应用实例
app = FastAPI()

# 获取当前目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 调试信息
uploads_dir = os.path.join(BASE_DIR, "uploads")
static_dir = os.path.join(BASE_DIR, "static")
print(f"BASE_DIR: {BASE_DIR}")
print(f"Uploads directory: {uploads_dir}, exists: {os.path.exists(uploads_dir)}")
print(f"Static directory: {static_dir}, exists: {os.path.exists(static_dir)}")

# 先挂载uploads目录
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")
# 再挂载静态文件目录
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    print("Static directory not found, skipping mount")

@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        "static_server:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )