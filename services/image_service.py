import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import io
import os
import uuid
import time
from typing import List, Tuple, Optional
# 加载环境变量
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

class ImageService:
    def __init__(self):
        # 初始化YOLO模型
        self.model = None
        self._load_model()
        
        # 从环境变量加载上传目录
        upload_folder = os.getenv("UPLOAD_FOLDER", "uploads")
        self.upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), upload_folder)
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def _load_model(self):
        """
        加载YOLO模型
        """
        try:
            # 尝试导入ultralytics的YOLO
            from ultralytics import YOLO
            
            # 从环境变量获取模型路径
            env_model_path = os.getenv("YOLO_MODEL_PATH", "models/best.pt")
            
            # 确保模型路径是绝对路径或相对于项目根目录
            if not os.path.isabs(env_model_path):
                # 相对于项目根目录的路径
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_path = os.path.join(base_dir, env_model_path)
            else:
                model_path = env_model_path
            
            # 如果指定的模型不存在，尝试默认位置
            if not os.path.exists(model_path):
                # 尝试默认位置
                default_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best.pt")
                if os.path.exists(default_model_path):
                    model_path = default_model_path
                else:
                    # 使用预训练模型
                    model_path = "yolov8n.pt"
            
            self.model = YOLO(model_path)
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            # 如果YOLO加载失败，使用一个简单的模拟检测函数
            self.model = None
    
    def process_image(self, image_data: bytes) -> np.ndarray:
        """
        处理图片数据，转换为OpenCV格式
        """
        # 读取图片
        img = Image.open(io.BytesIO(image_data))
        
        # 转换为RGB格式
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # 转换为OpenCV格式
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        return img_cv
    
    def detect_number_plates(self, img: np.ndarray) -> List[Tuple[float, float, float, float, float, int]]:
        """
        使用YOLO模型检测车牌
        返回格式: [(x1, y1, x2, y2, confidence, class_id), ...]
        """
        if self.model is None:
            # 如果模型未加载，使用模拟检测
            return self._mock_detection(img)
        
        try:
            # 进行检测
            results = self.model(img)  # 不指定类别，让模型检测所有类别
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # 获取边界框坐标和置信度
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = box.cls[0].item()
                    
                    # 确保坐标在有效范围内
                    h, w = img.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    detections.append((x1, y1, x2, y2, conf, int(cls)))
            
            # 按置信度排序，返回前N个结果
            detections.sort(key=lambda x: x[4], reverse=True)
            return detections[:5]  # 返回前5个最可能的结果
            
        except Exception as e:
            print(f"检测过程中出错: {str(e)}")
            # 如果检测失败，返回模拟结果
            return self._mock_detection(img)
    
    def _mock_detection(self, img: np.ndarray) -> List[Tuple[float, float, float, float, float, int]]:
        """
        模拟检测结果（用于测试）
        """
        h, w = img.shape[:2]
        # 返回一个模拟的车牌区域
        return [(w*0.3, h*0.6, w*0.7, h*0.75, 0.85, 0)]
    
    def draw_detections(self, img: np.ndarray, detections: List, results: List) -> np.ndarray:
        """
        在图像上绘制检测结果
        使用PIL来渲染中文文字，确保中文正确显示
        """
        result_img = img.copy()
        
        # 对于每个检测结果
        for i, (detection, result) in enumerate(zip(detections, results)):
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 绘制边界框
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 准备标签文本
            plate_text = result['number_plate']
            
            try:
                # 使用PIL来渲染中文文字
                # 将OpenCV图像转换为PIL图像
                res_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(res_pil)
                
                # 尝试加载Windows系统中的中文字体
                font_scale = max(12, int((y2 - y1) * 0.3))  # 根据车牌高度动态调整字体大小
                fonts = [
                    "simhei.ttf",  # 黑体
                    "simsun.ttc",  # 宋体
                    "msyh.ttc",    # 微软雅黑
                    "Arial.ttf"
                ]
                font = None
                
                # 尝试找到可用的字体
                for font_name in fonts:
                    try:
                        # 尝试从系统字体目录加载
                        font = ImageFont.truetype(font_name, font_scale)
                        break
                    except:
                        try:
                            # 尝试完整路径
                            font = ImageFont.truetype(f"C:\\Windows\\Fonts\\{font_name}", font_scale)
                            break
                        except:
                            continue
                
                # 如果找不到字体，使用默认字体
                if font is None:
                    font = ImageFont.load_default()
                    
                # 文字位置在边界框上方，如果上方空间不足则放在下方
                text_position = (x1, max(10, y1 - font_scale - 5))
                if y1 < font_scale + 10:  # 如果上方空间不足
                    text_position = (x1, min(img.shape[0] - 5, y2 + font_scale + 5))
                
                # 绘制文字（PIL使用RGB颜色）
                draw.text(text_position, plate_text, font=font, fill=(0, 255, 0))
                
                # 如果置信度大于0.1，也显示置信度
                if conf > 0.1:
                    confidence_text = f"置信度: {conf:.2f}"
                    conf_position = (x1, text_position[1] + font_scale + 2)
                    draw.text(conf_position, confidence_text, font=font, fill=(0, 255, 0))
                
                # 将PIL图像转换回OpenCV格式
                result_img = cv2.cvtColor(np.array(res_pil), cv2.COLOR_RGB2BGR)
                
            except Exception as e:
                print(f"绘制文字时出错: {e}")
                # 回退到cv2的putText方法
                label = f"{plate_text}: {conf:.2f}"
                cv2.putText(
                    result_img, 
                    label, 
                    (x1, max(10, y1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
        
        return result_img
    
    def save_image(self, img: np.ndarray, image_type: str) -> str:
        """
        保存图像并返回相对路径
        """
        # 生成唯一文件名
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{image_type}_{timestamp}_{unique_id}.jpg"
        
        # 保存图像
        save_path = os.path.join(self.upload_dir, filename)
        cv2.imwrite(save_path, img)
        
        # 返回相对路径
        return f"/uploads/{filename}"
    
    def save_plate_region(self, img: np.ndarray, coordinates: tuple) -> str:
        """
        从原始图像中截取车牌区域并保存
        
        Args:
            img: 原始图像
            coordinates: 车牌区域坐标 (x1, y1, x2, y2)
            
        Returns:
            保存的车牌区域图片的相对路径
        """
        x1, y1, x2, y2 = coordinates
        # 确保坐标为整数
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 截取车牌区域
        roi = img[y1:y2, x1:x2]
        
        # 如果截取区域为空，返回None
        if roi.size == 0:
            return None
        
        # 生成唯一文件名
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        filename = f"plate_region_{timestamp}_{unique_id}.jpg"
        
        # 保存图像
        save_path = os.path.join(self.upload_dir, filename)
        cv2.imwrite(save_path, roi)
        
        # 返回相对路径
        return f"/uploads/{filename}"