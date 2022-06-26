import os

def main():
    videos_dir = "/nas/database/DISFA_Database/Videos_LeftCamera/"
    output_dir = '/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/DISFA/frames'

    for f in os.listdir(videos_dir):
        if (f.rsplit(".")[-1] == "avi"):
            print(f"Processing {f}")
            out_path = os.path.join(output_dir, f[9:14])
            os.makedirs(out_path, exist_ok=True)
            command = "ffmpeg -i " + os.path.join(videos_dir, f) + " " + os.path.join(out_path, "%d.png") 
            os.system(command)

if __name__ == "__main__":
    main()