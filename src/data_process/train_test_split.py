import random
import os


dataset_dir = '../../dataset/a2d2'

dataset_txt_path = os.path.join(dataset_dir, 'dataset.txt')
train_file = os.path.join(dataset_dir, 'train.txt')
val_file = os.path.join(dataset_dir, 'val.txt')
test_file = os.path.join(dataset_dir, 'test.txt')

filenames = [x for x in open(dataset_txt_path).readlines()]
print(filenames[0:5])
print('The total number of dataset sampels : %d' % len(filenames)) 
filenames.sort()  # make sure that the filenames have a fixed order before shuffling
random.seed(230)
random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)

split_1 = int(0.85 * len(filenames)) 
split_2 = int(0.95 * len(filenames))

train_file_list = filenames[:split_1]  
print('The number of train files : %d' % len(train_file_list))      
val_file_list = filenames[split_1:split_2]
print('The number of validation files : %d' % len(val_file_list))    
test_file_list = filenames[split_2:]
print('The number of test files : %d' % len(test_file_list))   

with open(train_file, 'w') as f:
    for line in train_file_list:
        f.write("%s" % line)

with open(val_file, 'w') as f:
    for line in val_file_list:
        f.write("%s" % line)

with open(test_file, 'w') as f:
    for line in test_file_list:
        f.write("%s" % line)
