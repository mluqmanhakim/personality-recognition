import os
import time

def main():
    data_list_path = '/nas/database/BP-4DSFE/ImageAnd3D_data/list.txt'
    data_dir = '/nas/database/BP-4DSFE/ImageAnd3D_data'
    output_dir = '/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/BP4D/img'

    with open(data_list_path) as f:
        lines = f.readlines()

    start_time = time.time()
    for line in lines:
        arr = line.strip('\n').split('\\')
        dir_1 = os.path.join(output_dir, arr[2])
        dir_2 = os.path.join(dir_1, arr[3])
        os.makedirs(dir_1, exist_ok=True)
        os.makedirs(dir_2, exist_ok=True)
        img_path = os.path.join(data_dir, arr[2], arr[3], arr[4])
        out_path = os.path.join(dir_2, arr[4])
        command = "cp " + img_path + " " + out_path
        os.system(command)
        
    print("--- %s seconds ---" % (time.time() - start_time))

    
if __name__ == "__main__":
    main()