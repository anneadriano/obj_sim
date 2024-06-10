import random


pos_list = [1,2,3,4,5,6,7,8,9,10]
num_frames = 5

int_range = range(0,len(pos_list)-num_frames)
start_index = random.choice(int_range)
print(start_index)