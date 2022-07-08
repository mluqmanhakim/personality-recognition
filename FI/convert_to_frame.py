import os
import pickle
import time

def main():
    start_time = time.time()
    video_dir = "/nas/database/Big5/train"
    output_dir = "/nas/project_data/B1_Behavior/hakim/FI/frames/train"
    annotation_path = '/nas/database/Big5/gt/annotation_training.pkl'
    with open(annotation_path, 'rb') as f:
        annotation = pickle.load(f, encoding='latin1')
    video_names = list(annotation['extraversion'].keys())

    for video_name in video_names:
        print(f'Processing {video_name}')
        dir_name = video_name.rsplit(".")[0] + "." + video_name.rsplit(".")[1] 
        out_path = os.path.join(output_dir, dir_name)
        os.makedirs(out_path, exist_ok=True)
        video_path = os.path.join(video_dir, video_name)
        command = "/home/luqman/ffmpeg/ffmpeg-git-20220622-amd64-static/ffmpeg -loglevel warning -t 15 -i " + video_path + " -r:v 30 -frames:v 450 " + os.path.join(out_path, "%d.png") 
        os.system(command)
    
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()