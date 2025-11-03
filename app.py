from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import time
from datetime import datetime, timedelta
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
import asyncio
import json
import torch
import sys
# 加载环境变量
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()
# 导入自定义模块
from controllers.image_controller import ImageController

print("=== 启动前检测 ===")
import paddle
print(f"PaddlePaddle版本: {paddle.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"OpenCV version: {cv2.__version__}")

from ultralytics import YOLO
print("=== 开始加载模型 ===")

# 从环境变量加载模型路径
model_path = os.getenv("YOLO_MODEL_PATH", "models/best.pt")
model = YOLO(model_path)
print("成功加载模型:", model_path)
try:
    # 创建一个空白图像进行测试
    import numpy as np
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = model(test_img, verbose=False)
    print("=== 推理测试成功 ===")
    print(f"检测到 {len(results[0].boxes)} 个目标")
except Exception as e:
    print(f"推理测试失败: {e}")
    sys.exit(1)

print("=== 所有测试通过，继续运行主程序 ===")

# 创建日志管理器
class LogManager:
    def __init__(self):
        # 存储活跃的WebSocket连接
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast_log(self, log_type: str, message: str, timestamp: Optional[str] = None):
        """
        广播日志信息给所有连接的客户端
        log_type: 日志类型 (info, progress, error, success)
        message: 日志消息
        timestamp: 时间戳（可选）
        """
        if not timestamp:
            timestamp = datetime.now().isoformat()
        
        log_data = {
            "type": log_type,
            "message": message,
            "timestamp": timestamp
        }
        
        # 发送给所有活跃连接
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(log_data)
            except Exception as e:
                # 记录断开的连接
                disconnected.append(connection)
        
        # 清理断开的连接
        for conn in disconnected:
            self.disconnect(conn)

# 创建日志管理器实例
log_manager = LogManager()

# 创建FastAPI应用实例
app = FastAPI(title="车牌识别服务", description="提供车牌识别相关接口")

# 从环境变量加载上传目录
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), UPLOAD_FOLDER)

async def cleanup_old_files(days: int = 5):
    """
    清理指定天数前的临时文件
    - **days**: 要保留的文件天数，默认5天
    """
    if not os.path.exists(UPLOADS_DIR):
        await log_manager.broadcast_log("info", f"清理任务: 上传目录 {UPLOADS_DIR} 不存在")
        return
    
    current_time = time.time()
    threshold = current_time - (days * 24 * 60 * 60)  # 5天前的时间戳
    files_to_delete = []
    
    # 找出所有需要删除的文件
    for filename in os.listdir(UPLOADS_DIR):
        file_path = os.path.join(UPLOADS_DIR, filename)
        if os.path.isfile(file_path):
            file_mtime = os.path.getmtime(file_path)
            if file_mtime < threshold:
                files_to_delete.append(file_path)
    
    # 删除文件
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            await log_manager.broadcast_log("error", f"清理文件失败 {file_path}: {str(e)}")
    
    if deleted_count > 0:
        await log_manager.broadcast_log("info", f"清理任务: 成功删除 {deleted_count} 个过期文件")

async def periodic_cleanup_task():
    """
    定期执行清理任务的后台任务
    """
    await log_manager.broadcast_log("info", "启动定时清理任务")
    while True:
        # 每天凌晨2点执行清理任务
        now = datetime.now()
        # 计算到下一个凌晨2点的时间
        target_time = (now + timedelta(days=1)).replace(hour=2, minute=0, second=0, microsecond=0)
        wait_seconds = (target_time - now).total_seconds()
        
        await log_manager.broadcast_log("info", f"下次清理时间: {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 等待到下一个执行时间
        await asyncio.sleep(wait_seconds)
        
        # 执行清理任务
        await log_manager.broadcast_log("info", "开始执行清理任务")
        await cleanup_old_files()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化控制器
image_controller = ImageController(log_manager)

@app.post("/api/recognize", response_model=Dict[str, Any])
async def recognize_number_plate(file: UploadFile = File(...)):
    """
    车牌识别接口
    - **file**: 要上传的图片文件
    """
    try:
        # 验证文件类型
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="只支持图片文件")
        
        # 读取图片文件
        contents = await file.read()
        
        # 发送开始识别日志
        await log_manager.broadcast_log("info", f"开始识别图片: {file.filename}")
        
        # 调用控制器进行识别
        result = await image_controller.recognize(contents)
        
        # 发送识别完成日志
        await log_manager.broadcast_log("success", "识别完成")
        
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        # 发送错误日志
        await log_manager.broadcast_log("error", f"识别失败: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket接口，用于实时推送日志信息
    """
    await log_manager.connect(websocket)
    try:
        # 发送连接成功消息
        await log_manager.broadcast_log("info", "客户端已连接")
        
        # 保持连接
        while True:
            # 接收消息（如果需要）
            await websocket.receive_text()
    except WebSocketDisconnect:
        log_manager.disconnect(websocket)
        await log_manager.broadcast_log("info", "客户端已断开连接")
    except Exception as e:
        log_manager.disconnect(websocket)
        await log_manager.broadcast_log("error", f"WebSocket错误: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """
    应用启动时执行的事件
    """
    # 确保上传目录存在
    if not os.path.exists(UPLOADS_DIR):
        os.makedirs(UPLOADS_DIR)
    
    # 启动定时清理任务
    asyncio.create_task(periodic_cleanup_task())
    
    # 启动时执行一次清理
    await asyncio.sleep(5)  # 等待服务启动稳定
    await cleanup_old_files()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )