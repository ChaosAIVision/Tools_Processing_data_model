
from ultralytics import YOLO
import cv2
from cvzone import cornerRect
import torch
import os


class Process_model():
    def __init__(self, input_folder_path, output_folder_path,weight_path):
        print("Bắt đầu quá trình xử lí model ...")
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.weight_path = weight_path
    
    def __detect_object_YOLO__(self):
        print('Bắt đầu quá trình đánh nhãn...')
        image_names = os.listdir(self.input_folder_path)
        model = YOLO(self.weight_path)
        for image_name in image_names:
            image_path = os.path.join(self.input_folder_path, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image,(640,640))
            results = model(image, classes= 0)[0]
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(self.output_folder_path,label_name)
            with open(label_path, 'w') as f:
                # Lặp qua các kết quả dự đoán
                for rs in results:
                    # Lấy tensor của hộp giới hạn (bounding box)
                    rs = results.boxes[0]

                    box = rs.xywh.cpu().numpy()

                    # Trích xuất thông tin của hộp giới hạn từ tensor
                    x_center, y_center, w, h = box[0]
                    label = int(rs.cls[0])
                    x_center_norm = x_center / image.shape[1]
                    y_center_norm = y_center / image.shape[0]
                    w_norm = w / image.shape[1]
                    h_norm = h / image.shape[0] 


                    # Ghi các giá trị vào tập tin
                    f.write(f"{label} {x_center_norm} {y_center_norm} {w_norm} {h_norm}\n")
                    print(f'Đã đánh label file {image_name} tại {self.output_folder_path}')

            




