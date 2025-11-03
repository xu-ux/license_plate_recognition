from typing import Dict, Any, Optional, List
import cv2
import numpy as np
import os
import uuid
import time
from PIL import Image
import io

# 导入服务层
from services.image_service import ImageService
from services.ocr_service import OCRService

class ImageController:
    def __init__(self, log_manager=None):
        # 初始化服务
        self.image_service = ImageService()
        self.ocr_service = OCRService(log_manager)
        self.log_manager = log_manager
        
    async def recognize(self, image_data: bytes) -> Dict[str, Any]:
        """
        处理图片识别请求
        """
        try:
            # 发送进度日志
            if self.log_manager:
                await self.log_manager.broadcast_log("progress", "开始处理图片数据", None)
            
            # 1. 处理图片数据
            img = self.image_service.process_image(image_data)
            
            if self.log_manager:
                await self.log_manager.broadcast_log("progress", "图片数据处理完成", None)
                await self.log_manager.broadcast_log("progress", "保存原始图片", None)
            
            # 2. 保存原始图片
            original_path = self.image_service.save_image(img, "original")
            
            if self.log_manager:
                await self.log_manager.broadcast_log("progress", "开始检测车牌", None)
            
            # 3. 使用YOLO模型检测车牌
            detected_objects = self.image_service.detect_number_plates(img)
            
            if not detected_objects:
                if self.log_manager:
                    await self.log_manager.broadcast_log("info", "未检测到车牌", None)
                return {
                    "number_plate": "未检测到车牌",
                    "confidence": 0.0,
                    "original_image": original_path,
                    "result_image": original_path,
                    "message": "未检测到车牌"
                }
            
            if self.log_manager:
                await self.log_manager.broadcast_log("success", f"检测到 {len(detected_objects)} 个车牌", None)
            
            # 4. 对每个检测到的车牌进行OCR识别
            results = []
            # 保存车牌区域图片路径
            plate_region_paths = []
            
            for i, obj in enumerate(detected_objects):
                x1, y1, x2, y2, conf, cls = obj
                
                if self.log_manager:
                    await self.log_manager.broadcast_log("progress", f"处理第 {i+1} 个车牌区域 (置信度: {conf:.2f})")
                
                # 提取车牌区域
                roi = img[int(y1):int(y2), int(x1):int(x2)]
                
                # 保存车牌区域图片
                plate_region_path = self.image_service.save_plate_region(img, (x1, y1, x2, y2))
                plate_region_paths.append(plate_region_path)
                
                if self.log_manager:
                    await self.log_manager.broadcast_log("progress", "开始OCR文字识别")
                
                # 进行OCR识别（不需要传递log_manager，因为已经在OCRService构造函数中设置）
                plate_text, ocr_confidence = self.ocr_service.recognize_text(roi)
                
                if self.log_manager:
                    await self.log_manager.broadcast_log("info", f"识别结果: {plate_text}, 置信度: {ocr_confidence:.4f}")
                
                result_item = {
                    "number_plate": plate_text,
                    "ocr_confidence": float(ocr_confidence),  # OCR的置信度
                    "yolo_confidence": float(conf),  # YOLO检测车牌区域的置信度
                    "coordinates": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    }
                }
                
                # 如果成功保存了车牌区域图片，添加到结果中
                if plate_region_path:
                    result_item["plate_region_image"] = plate_region_path
                
                results.append(result_item)
            
            if self.log_manager:
                await self.log_manager.broadcast_log("progress", "绘制检测结果")
            
            # 5. 在原图上绘制检测结果
            result_img = self.image_service.draw_detections(img, detected_objects, results)
            
            if self.log_manager:
                await self.log_manager.broadcast_log("progress", "保存结果图片")
            
            # 6. 保存结果图片
            result_path = self.image_service.save_image(result_img, "result")
            
            if self.log_manager:
                await self.log_manager.broadcast_log("success", "识别完成")
            
            # 7. 返回识别结果
            return {
                "results": results,
                "original_image": original_path,
                "result_image": result_path,
                "message": "识别成功",
                # 添加车牌区域图片路径，只返回有效的路径
                "plate_region_images": [path for path in plate_region_paths if path is not None]
            }
            
        except Exception as e:
            error_msg = f"识别过程中出错: {str(e)}"
            print(error_msg)
            if self.log_manager:
                await self.log_manager.broadcast_log("error", error_msg)
            raise Exception(f"识别失败: {str(e)}")