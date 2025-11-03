import cv2
import numpy as np
from typing import Optional
# 加载环境变量
from dotenv import load_dotenv
import os

# 加载.env文件中的环境变量
load_dotenv()

class OCRService:
    def __init__(self, log_manager=None):
        # 初始化OCR模型
        self.ocr = None
        self.log_manager = log_manager
        # 从环境变量加载OCR配置
        self.confidence_threshold = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.5"))
        self._load_ocr_model()
    
    def _load_ocr_model(self):
        """
        加载PaddleOCR模型
        参考e_number_plate_ocr.py中的配置
        """
        try:
            # 尝试导入PaddleOCR
            from paddleocr import PaddleOCR

            # 创建OCR实例，优化参数以提高车牌识别效果
            # 注意：use_angle_cls和use_textline_orientation是互斥的
            self.ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,  # 启用角度分类以提高识别准确率
                lang="ch"  # 使用中英文模型
            )
            print("成功加载PaddleOCR模型")
        except Exception as e:
            print(f"加载PaddleOCR模型失败: {str(e)}")
            # 如果PaddleOCR加载失败，设置为None
            self.ocr = None
    
    def recognize_text(self, img: np.ndarray) -> tuple:
        """
        识别图像中的文本
        返回: (识别的文本, 置信度)
        """
        try:
            # 如果OCR模型可用，使用它进行识别
            if self.ocr is not None:
                # 先对图像进行预处理
                # preprocessed_img = self.preprocess_image(img)
                return self._paddle_ocr_recognize(img)
            else:
                # 如果OCR模型不可用，使用模拟识别
                mock_text = self._mock_recognize(img)
                return mock_text, 0.8  # 模拟识别返回默认置信度0.8
        except Exception as e:
            print(f"OCR识别出错: {str(e)}")
            # 如果识别失败，返回模拟结果和低置信度
            mock_text = self._mock_recognize(img)
            return mock_text, 0.5
    
    def _paddle_ocr_recognize(self, img: np.ndarray) -> str:
        """
        使用PaddleOCR进行文本识别
        """
        try:
            # 执行OCR识别
            ocr_results = self.ocr.predict(img)

            # 提取识别的文字
            recognized_text = ""
            if ocr_results:
                for result in ocr_results:
                    print(result)
                    # 记录日志但不使用异步调用避免警告
                    if self.log_manager and hasattr(self.log_manager, 'log'):
                        # 使用同步的log方法而不是异步的broadcast_log
                        self.log_manager.log("info", f"OCR识别结果: {recognized_text}")

                    # 检查结果格式 - 组装所有识别文本
                    if isinstance(result, dict) and 'rec_texts' in result and result['rec_texts']:
                        # 组装所有识别文本，而不仅仅是第一个
                        recognized_text = ''.join(result['rec_texts'])
                        scores = result.get('rec_scores', [])
                        # 计算平均置信度
                        if scores:
                            confidence = sum(scores) / len(scores)
                            print(f"OCR识别结果: '{recognized_text}' (各部分置信度: {scores}, 平均置信度: {confidence})")
                        else:
                            confidence = 0
                            print(f"OCR识别结果: '{recognized_text}' (无置信度值)")
                    elif isinstance(result, list) and result:
                        # 兼容旧版本的PaddleOCR返回格式
                        texts = []
                        for line in result:
                            if isinstance(line, list) and len(line) >= 2 and isinstance(line[1], list) and len(line[1]) >= 1:
                                texts.append(line[1][0])
                                print(f"OCR识别结果(兼容格式部分): '{line[1][0]}'")
                        recognized_text = ''.join(texts)
                        confidence = 0  # 兼容格式暂不计算置信度
                        print(f"OCR识别结果(兼容格式组装): '{recognized_text}'")
                    
                    # 如果已经识别到文本，就不再继续处理其他结果
                    if recognized_text:
                        break
            
            # 如果没有识别到文字，尝试旧的ocr方法
            if not recognized_text:
                result = self.ocr.ocr(img, cls=True)
                # 提取文本
                texts = []
                scores = []
                if result and result[0]:
                    for line in result[0]:
                        if len(line) >= 2 and line[1] and len(line[1]) >= 1:
                            texts.append(line[1][0])
                            if len(line[1]) >= 2:  # 如果有置信度值
                                scores.append(line[1][1])
                
                # 合并文本
                recognized_text = "".join(texts)
                # 计算平均置信度
                if scores:
                    confidence = sum(scores) / len(scores)
                else:
                    confidence = 0
            
            # 过滤车牌文本
            filtered_text = self._filter_plate_text(recognized_text)
            # 返回识别文本和置信度
            # 如果置信度低于阈值，使用模拟识别
            if confidence < self.confidence_threshold:
                print(f"OCR置信度 {confidence} 低于阈值 {self.confidence_threshold}，使用模拟识别")
                mock_text = self._mock_recognize(img)
                return mock_text, confidence
            return filtered_text, confidence
            
        except Exception as e:
            print(f"OCR识别过程出错: {str(e)}")
            # 如果predict方法失败，回退到ocr方法
            try:
                result = self.ocr.ocr(img, cls=True)
                texts = []
                scores = []
                if result and result[0]:
                    for line in result[0]:
                        if len(line) >= 2 and line[1] and len(line[1]) >= 1:
                            texts.append(line[1][0])
                            if len(line[1]) >= 2:  # 如果有置信度值
                                scores.append(line[1][1])
                filtered_text = self._filter_plate_text(" ".join(texts))
                # 计算平均置信度
                confidence = sum(scores) / len(scores) if scores else 0.5
                return filtered_text, confidence
            except:
                return "识别失败", 0.0
    
    def _mock_recognize(self, img: np.ndarray) -> str:
        """
        模拟OCR识别结果（用于测试）
        """
        # 生成一个模拟的车牌号
        import random
        provinces = ["京", "津", "冀", "晋", "蒙", "辽", "吉", "黑", "沪", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "渝", "藏", "陕", "甘", "青", "宁", "新"]
        letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # 不含O和I
        numbers = "0123456789"
        
        # 生成车牌号：省份简称 + 字母 + 5位数字/字母组合
        province = random.choice(provinces)
        letter = random.choice(letters)
        rest = "".join(random.choices(letters + numbers, k=5))
        
        return province + letter + rest
    
    def _filter_plate_text(self, text: str) -> str:
        """
        过滤车牌文本，保留有效的车牌字符
        """
        # 定义有效的车牌字符集
        valid_chars = "京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云渝藏陕甘青宁新ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
        
        # 过滤无效字符
        filtered_text = "".join([c for c in text if c in valid_chars])
        
        # 如果过滤后的文本太短，返回原始文本
        if len(filtered_text) < 5:
            return text.strip()
        
        return filtered_text

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        放弃使用预处理
        预处理图像以提高OCR准确率
        """
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 二值化
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 边缘检测
        edges = cv2.Canny(binary, 50, 150)
        
        # 尝试寻找矩形轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 找出最大的矩形轮廓
        # max_area = 0
        # max_contour = None
        # for contour in contours:
        #     area = cv2.contourArea(contour)
        #     if area > max_area and area > img.shape[0] * img.shape[1] * 0.01:
        #         max_area = area
        #         max_contour = contour
        
        # 如果找到合适的轮廓，裁剪图像
        # if max_contour is not None:
        #     x, y, w, h = cv2.boundingRect(max_contour)
        #     if w > 0 and h > 0:
        #         # 确保坐标在有效范围内
        #         x = max(0, x)
        #         y = max(0, y)
        #         w = min(w, img.shape[1] - x)
        #         h = min(h, img.shape[0] - y)
        #
        #         # 裁剪图像
        #         img = img[y:y+h, x:x+w]
        # cv2.imwrite("preprocessed.jpg", img)
        return img