from process_data import Process_data
from process_model import Process_model

nothing = '[None]'
in_path = '[input_folder_path]'
out_path ='[output_folder_path]'
in_out = '[input_folder_path & output_folder_path]'

if __name__ =='__main__':
    tmp = 0
    while tmp != 'done':
        print( '__' * 20)
        print('Nhập option cần processing  vào đây \n 1: data \n 2: model \ndone : Kết thúc')
        tmp = input('Nhập option: ')
        if tmp == 'done':
            print('Kết thúc quá trình processing !')
        
        if tmp == '1':
            print( '__' * 20)
            print("MENU")
            print(f'1: Kiểm tra images và labels có file tương đồng hay không {out_path}')
            print(f'2: Kiểm tra xem có file_labels nào rỗng hay không {in_out}')
            print(f'3: Tách ảnh ra khỏi folder gộp chung {in_out}')
            print(f'4: Xóa nhãn labels dư thừa {in_out} ')
            print(f'5: Chuyển đổi nhãn trong file labels {in_path}')
            print(f'6: Viết file YAML để train model yolo {out_path}')
            print(f'7: Chia tập train, dev, test {in_out} ')
            print(f'8: Hiển thị ảnh kèm boudingbox {nothing}')
            print(f'9: Cắt frames từ video {out_path}')
            print('0: Kết thúc')
            
            
          
            print( '__' * 20)
            data = input('Chọn option của data processing: ')

            if data == '0':
                print('Kết thúc quá trình processing !')
                break
                #Khai báo class Process_data
            input_folder_path = input('Nhập input_folder_ path: ')
            output_folder_path = input('Nhập output_folder_path: ')
            data_process = Process_data(input_folder_path,output_folder_path)
            
            if data == '1':
                print( '__' * 20)
                images_folder_path = input("Nhập path images cần kiểm tra: ")
                labels_folder_path = input('Nhập path labels folders cần kiểm tra: ')
                data_process.__check_img_labels__(images_folder_path,labels_folder_path)
            
            if data == '2':
                print( '__' * 20)
                data_process.__remove_empty_labels__()
            
            if data == '3':
                print( '__' * 20)
                data_process.__separate__images_labels__()

            if data == '4':
                print( '__' * 20)
                data_process.__remove_object_labels__()
            if data == '5':
                print( '__' * 20)
                print(" Nhập vào theo định dạng {'key1' : value1 , 'key2': value2}")
                label_mapping = input(' Nhập labels mapping: ')
                data_process.__change_labels_in_folder__(label_mapping) 
            if data == '6':
                print( '__' * 20)
                filename = input('Nhập tên file YAML: ')
                train_image_dir =  input('Nhập path dẫn đến folder chứa ảnh tập train: ')
                val_image_dir =  input('Nhập path dẫn đến folder chứa ảnh tập val: ')
                classes = input('Nhập classes theo định dạng ["obj1" , "obj2"]: ')
                data_process.__write_file_YAML__(filename,train_image_dir,val_image_dir, classes)

            if data == '7':
                print( '__' * 20)
                print( 'Tổng train + dev + test = 1')
                train = input('Nhập tỉ lệ train: ')
                dev = input('Nhập tỉ lệ dev: ')
                test = input("Nhập tỉ lệ test: ")
                seed = input('Nhập seed: ')
                data_process.__split_folder__(train,dev,test,seed)


            if data == '8':
                print( '__' * 20)
                image_folder_path = input("Nhập path folder ảnh: ")
                label_folder_path = input('Nhập path folder labels: ')
                num_img_per_row = input('Nhập số lượng ảnh trên 1 hàng: ')
                num_rows = input("Nhập số hàng: ")
                data_process.__show_bounding_box__(image_folder_path,label_folder_path,num_img_per_row,num_rows)

            
            if data == '9':
                print( '__' * 20)
                image_video_path = input('Nhập path video vào đây:')
                num_frames = input('Nhập số 2frames cần lấy vào đây: VD num_frames = 5\nthì cắt ra 5 frames sẽ lưu lại 1 frames: ')
                data_process.__cut_frames_from_video__(image_video_path,num_frames)
        
        if tmp == '2':
            print( '__' * 20)
            print("MENU")
            print(f'1: Đánh nhãn tự động với pretrain weight của YOLOv8 {in_out}')

            model = input('Chọn option của model processing: ')

            if model == '0':
                print('Kết thúc quá trình processing !')
                break

            input_folder_path = input('Nhập input_folder_ path: ')
            output_folder_path = input('Nhập output_folder_path: ')
            weight_path = input('Nhập weight_path: ')
            model_process = Process_model(input_folder_path,output_folder_path,weight_path)

            if model == '1':
                print( '__' * 20)
                model_process.__detect_object_YOLO__()







                


# /mnt/d/Nguyen_Nhat_Truong/Guild_learn_AI/AL_ML/Deep_learning_vietnguyen/module/helmet_train/weights/best.pt