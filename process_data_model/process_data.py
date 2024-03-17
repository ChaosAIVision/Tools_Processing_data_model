import os
import ast
import shutil
import json
import splitfolders
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Process_data():
    def __init__(self,input_folder_path, output_folder_path):
        print('bắt đầu quá trình process data...')
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path

    def __check_img_labels__(self, images_folder_path, labels_folder_path):
        print('bắt đầu quá trình check_img_labels...')      
        images_files = os.listdir(images_folder_path)
        label_files = os.listdir(labels_folder_path)
        
        images_names = {os.path.splitext(file)[0] for file in images_files}
        labels_names = {os.path.splitext(file)[0] for file in label_files}
        
        missing_image_names = images_names - labels_names
        missing_label_names = labels_names - images_names

        if missing_image_names:
            print('Các ảnh không có nhãn tương đồng là:')
            for image_name in missing_image_names:
                print(image_name)
                shutil.move(os.path.join(images_folder_path, f"{image_name}.jpg"), os.path.join( self.output_folder_path, f"{image_name}.jpg"))
                print(f'Đã chuyển ảnh thiếu qua {self.output_folder_path}')
        if missing_label_names:
            print('Các labels không có ảnh tương ứng là:')
            for label_name in missing_label_names:
                print(label_name)
                shutil.move(os.path.join(labels_folder_path, f"{label_name}.txt"), os.path.join(self.output_folder_path, f"{label_name}.txt"))
                print(f' Đã chuyển nhãn  thiếu qua {self.output_folder_path}')


        if not (missing_image_names or missing_label_names):
            print('Không có ảnh hoặc labels bị thiếu ')

    def __remove_empty_labels__(self):
        print('Bắt đầu quá trình remove_empty_labels...')
        for filename in os.listdir(self.input_folder_path):
            file_path = os.path.join(self.input_folder_path, filename)
            if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
                # Nếu là file rỗng, di chuyển nó sang thư mục đích
                shutil.move(file_path, os.path.join(self.output_folder_path, filename))
                print(" Dã chuyển:", filename)

    def __separate__images_labels__(self):
        print('Bắt đầu quá trình separate_images_labels')
        for filename in os.listdir( self.input_folder_path):
            images_path = os.path.join(self.input_folder_path,filename)
            if os.path.isfile(images_path) and filename.lower().endswith('.jpg'):
                output_path = os.path.join(self.output_folder_path,filename)
                shutil.move(images_path, output_path)
                print(f' đã chuyển  { filename} đến {output_path}')

    def __remove_object_labels__(self):
        
        label_keep = []
        while True:
            print( '__' * 20)           
            print('Bắt đầu quá trình nhập labels cần giữ ... Nhập break để kết thúc nếu đã đủ')
            tmp = input('Nhập index labels cần giữ: ')
            if tmp == 'break':
                break
            else:
                label_keep.append(int(tmp))
        print('Bắt đầu quá trình remove_object_labels')

    
        for filename in os.listdir(self.input_folder_path):
            if filename.endswith(".txt"):
                input_file_path = os.path.join(self.input_folder_path, filename)
                output_file_path = os.path.join(self.output_folder_path, filename)

                with open(input_file_path, 'r') as f:
                    lines = f.readlines()

                filtered_lines = []
                for line in lines:
                    if line.strip():  # Kiểm tra xem dòng không phải là dòng trống
                        label = int(line.split()[0])
                        if label in label_keep:
                            filtered_lines.append(line)

                with open(output_file_path, 'w') as f:
                    f.writelines(filtered_lines)


    def __change_labels_in_folder__(self, label_mapping ):
        print('Bắt đầu quá trình change_labels_in_folder...')
        label_mapping = ast.literal_eval(label_mapping)
         
        for filename in os.listdir(self.input_folder_path):
            file_path = os.path.join(self.input_folder_path, filename)
            if os.path.isfile(file_path) and filename.endswith('.txt'):
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                with open(file_path, 'w') as file:
                    for line in lines:
                        parts = line.split()
                        if parts:  # Kiểm tra xem dòng có chứa thông tin không
                            label = parts[0]
                            if label in label_mapping:
                                parts[0] = str(label_mapping[label])
                            else:
                                print(f"Warning: Label {label} not found in label_mapping. Skipping...")
                            file.write(' '.join(parts) + '\n')

    def __write_file_YAML__(self,yaml_filename, train_image_dir,val_image_dir, classes):
        print("Bắt đầu quá trình write_file_YAM:L")
        classes = json.loads(classes)        
        yaml_path = os.path.join(self.output_folder_path,yaml_filename)
        with open(yaml_path, "w") as f:
            f.write("train: " + os.path.abspath(train_image_dir) + "\n")
            f.write("val: " + os.path.abspath(val_image_dir) + "\n")
            f.write("nc: " + str(len(classes)) + "\n")
            f.write("names: " + str(classes) + "\n")
        print(f'Đã tạo file YAML tại {self.output_folder_path}')
        return yaml_path
    

    def __split_folder__(self,train,dev,test,seed):
        print('Bắt đầu quá trình split_folder...')
        train, dev, test, seed = float(train), float(dev), float(test), float(seed)
        splitfolders.ratio(self.input_folder_path,self.output_folder_path,seed=seed, ratio= (train,dev,test),group_prefix=None)
        print(f'Tập dữ liệu chia thành train : {train}, dev" {dev}, test{test} tại {self.output_folder_path}')


    def __show_bounding_box__(self,images_folder_path, label_folder_path, num_img_per_row, num_rows):
        print('Bắt đầu quá trình show_bouding_box')
        num_img_per_row = int(num_img_per_row)
        num_rows = int(num_rows)
        image_file_names = [file for file in os.listdir(images_folder_path)]
        selected_images = random.sample(image_file_names, num_img_per_row * num_rows)

        # Khởi tạo biến đếm
        images_displayed_count = 0

        # Tạo subplot với chiều ngang và chiều cao tương ứng
        fig, axs = plt.subplots(num_rows, num_img_per_row, figsize=(15, 10))

        for row in range(num_rows):
            for col in range(num_img_per_row):
                if images_displayed_count >= len(selected_images):
                    break

                img_file = selected_images[images_displayed_count]
                img_path = os.path.join(images_folder_path, img_file)
                label_file = img_file.replace('.jpg', '.txt')
                label_path = os.path.join(label_folder_path, label_file)

                # Đọc ảnh
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                with open(label_path, 'r') as f:
                    lines = f.readlines()

                # Nếu có bounding box, tăng biến đếm và hiển thị
                if len(lines) > 0:
                    images_displayed_count += 1

                    axs[row, col].imshow(image)

                    for line in lines:
                        parts = line.strip().split()
                        classed, x, y, w, h = map(float, parts)
                        x, y, width, height = (
                            x * image.shape[1],
                            y * image.shape[0],
                            w * image.shape[1],
                            h * image.shape[0]
                        )
                        classed = int(classed)

                        # Chọn màu sắc dựa trên lớp của bounding box
                        color = 'r' if classed == 1 else 'y'  # Màu đỏ cho lớp 1, màu xanh cho lớp 2 (tùy chỉnh theo yêu cầu)

                        # Tạo bounding box
                        rect = patches.Rectangle(
                            (x - width/2, y - height/2), width, height,
                            linewidth=1, edgecolor=color, facecolor='None'
                        )

                        # Thêm bounding box vào hình ảnh
                        axs[row, col].add_patch(rect)

                        # Hiển thị nhãn của bounding box
                        axs[row, col].text(x - width/2, y - height/2, f'Class {classed}', color=color)

        # Hiển thị tất cả các bounding box
        plt.tight_layout()
        plt.show()

    def __cut_frames_from_video__(self, video_path, num_frames):
        print('Bắt đầu quá trình cut_frames_from_video')
        # Tạo thư mục đầu ra nếu nó chưa tồn tại
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)

        vidcap = cv2.VideoCapture(video_path)
        
        # Initialize variables
        success, image = vidcap.read()
        count = 0
        num_frames = int(num_frames)  # Save every num_frames-th frame

        while success:
            if count % num_frames == 0:  # Check if it's time to save a frame
                if count == 0:
                    cv2.imwrite(os.path.join(self.output_folder_path, "frame%d.jpg" % count), image)  # Save frame as JPEG file
                    print('Lưu frames %d' % count, f'Tại {self.output_folder_path}')
                else:
                    count_v2 = count // num_frames  # chia cho ba để in frames liên tiếp
                    cv2.imwrite(os.path.join(self.output_folder_path, "frame%d.jpg" % count_v2), image)  # Save frame as JPEG file
                    print('Lưu frames %d' % count_v2, f'Tại {self.output_folder_path}' )
                    count_v2 = count
            success, image = vidcap.read()  # Read the next frame
            count += 1

        vidcap.release()  # Release the video capture object

