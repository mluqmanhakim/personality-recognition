import os
import numpy as np
import pandas as pd

# fold1:  train : part1+part2 test: part3
# fold2:  train : part1+part3 test: part2
# fold3:  train : part2+part3 test: part1

list_path_prefix = '../data/DISFA/list/'

# BP4D
BP4D_Sequence_split = [['F001', 'M007', 'F018', 'F008', 'F002', 'M004', 'F010', 'F009', 'M012', 'M001', 'F016', 'M014', 'F023', 'M008'],
					   ['M011', 'F003', 'M010', 'M002', 'F005', 'F022', 'M018', 'M017', 'F013', 'M016', 'F020', 'F011', 'M013', 'M005'],
					   ['F007', 'F015', 'F006', 'F019', 'M006', 'M009', 'F012', 'M003', 'F004', 'F021', 'F017', 'M015', 'F014']]

tasks = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
label_folder = '/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/BP4D/AU_labels'

def get_AUlabels_BP4D(seq, task, path):
	path_label = os.path.join(path, '{sequence}_{task}.csv'.format(sequence=seq, task=task))
	usecols = ['0', '1', '2', '4', '6', '9', '12', '15', '17']
	df = pd.read_csv(path_label, header=0, index_col=0, usecols=usecols)
	frames = [str(item) for item in list(df.index.values)]
	frames_path = ['{}/{}/{}'.format(seq, task, item) for item in frames]
	labels = df.values
	return frames_path, labels


####################################################################################################

# BP4D TEST FOLD 3
print("Processing BP4D TEST FOLD 3")
with open(list_path_prefix + 'test_img_path_fold3.txt','w') as f:
    u = 0

sequences = BP4D_Sequence_split[0]
frames = None
labels = None
len_f = 0
for seq in sequences:
	for t in tasks:
		temp_frames, temp_labels = get_AUlabels_BP4D(seq, t, label_folder)
		if frames is None:
			labels = temp_labels
			frames = temp_frames  # str list
		else:
			labels = np.concatenate((labels, temp_labels), axis=0)  # np.ndarray
			frames = frames + temp_frames  # str list
BP4D_image_path_list_part1 = frames
BP4D_image_label_part1 = labels

for frame in BP4D_image_path_list_part1:
	frame_img_name = frame + '.jpg'
	with open(list_path_prefix + 'test_img_path_fold3.txt', 'a+') as f:
		f.write(frame_img_name + '\n')

np.savetxt(list_path_prefix + 'test_label_fold3.txt', BP4D_image_label_part1 ,fmt='%d', delimiter=' ')

####################################################################################################

# BP4D TEST FOLD 2
print("Processing BP4D TEST FOLD 2")
with open(list_path_prefix + 'test_img_path_fold2.txt','w') as f:
    u = 0

sequences = BP4D_Sequence_split[1]
frames = None
labels = None
len_f = 0
for seq in sequences:
	for t in tasks:
		temp_frames, temp_labels = get_AUlabels_BP4D(seq, t, label_folder)
		if frames is None:
			labels = temp_labels
			frames = temp_frames  # str list
		else:
			labels = np.concatenate((labels, temp_labels), axis=0)  # np.ndarray
			frames = frames + temp_frames  # str list
BP4D_image_path_list_part2 = frames
BP4D_image_label_part2 = labels

for frame in BP4D_image_path_list_part2:
	frame_img_name = frame + '.jpg'
	with open(list_path_prefix + 'test_img_path_fold2.txt', 'a+') as f:
		f.write(frame_img_name + '\n')

np.savetxt(list_path_prefix + 'test_label_fold2.txt', BP4D_image_label_part2, fmt='%d', delimiter=' ')

####################################################################################################

# BP4D TEST FOLD 1
print("Processing BP4D TEST FOLD 1")
with open(list_path_prefix + 'test_img_path_fold1.txt','w') as f:
    u = 0

sequences = BP4D_Sequence_split[2]
frames = None
labels = None
len_f = 0
for seq in sequences:
	for t in tasks:
		temp_frames, temp_labels = get_AUlabels_BP4D(seq, t, label_folder)
		if frames is None:
			labels = temp_labels
			frames = temp_frames  # str list
		else:
			labels = np.concatenate((labels, temp_labels), axis=0)  # np.ndarray
			frames = frames + temp_frames  # str list
BP4D_image_path_list_part3 = frames
BP4D_image_label_part3 = labels

for frame in BP4D_image_path_list_part3:
	frame_img_name = frame + '.jpg'
	with open(list_path_prefix + 'test_img_path_fold1.txt', 'a+') as f:
		f.write(frame_img_name + '\n')

np.savetxt(list_path_prefix + 'test_label_fold1.txt', BP4D_image_label_part3 ,fmt='%d', delimiter=' ')

####################################################################################################

# TRAIN FOLD 1
print("Processing TRAIN FOLD 1")
with open(list_path_prefix + 'train_img_path_fold1.txt','w') as f:
    u = 0

train_img_label_fold1_list = BP4D_image_path_list_part1 + BP4D_image_path_list_part2
for frame in train_img_label_fold1_list:
	frame_img_name = frame + '.jpg'
	with open(list_path_prefix + 'train_img_path_fold1.txt', 'a+') as f:
		f.write(frame_img_name + '\n')

train_img_label_fold1_numpy_list = np.concatenate((BP4D_image_label_part1, BP4D_image_label_part2), axis=0)

np.savetxt(list_path_prefix + 'train_label_fold1.txt', train_img_label_fold1_numpy_list, fmt='%d')

#################################################################################

# TRAIN FOLD 2
print("Processing TRAIN FOLD 2")
with open(list_path_prefix + 'train_img_path_fold2.txt','w') as f:
    u = 0

train_img_label_fold2_list = BP4D_image_path_list_part1 + BP4D_image_path_list_part3
for frame in train_img_label_fold2_list:
	frame_img_name = frame + '.jpg'
	with open(list_path_prefix + 'train_img_path_fold2.txt', 'a+') as f:
		f.write(frame_img_name + '\n')

train_img_label_fold2_numpy_list = np.concatenate((BP4D_image_label_part1, BP4D_image_label_part3), axis=0)

np.savetxt(list_path_prefix + 'train_label_fold2.txt', train_img_label_fold2_numpy_list, fmt='%d')

#################################################################################

# TRAIN FOLD 3
print("Processing TRAIN FOLD 3")
with open(list_path_prefix + 'train_img_path_fold3.txt','w') as f:
    u = 0

train_img_label_fold3_list = BP4D_image_path_list_part2 + BP4D_image_path_list_part3
for frame in train_img_label_fold3_list:
	frame_img_name = frame + '.jpg'
	with open(list_path_prefix + 'train_img_path_fold3.txt', 'a+') as f:
		f.write(frame_img_name + '\n')

train_img_label_fold3_numpy_list = np.concatenate((BP4D_image_label_part2, BP4D_image_label_part3), axis=0)
np.savetxt(list_path_prefix + 'train_label_fold3.txt', train_img_label_fold3_numpy_list, fmt='%d')
