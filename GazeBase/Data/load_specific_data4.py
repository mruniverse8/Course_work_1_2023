import random
import os
def writein(lst, fdval):
    for txt in lst:
        fdval.write(txt)
fd4 = open("train_data.txt","w")
fd5 = open("test_data.txt","w")
fd6 = open("val_data.txt","w")
lst = []
folder = "./MERGED_SPECIFIC_DATA"
if not os.path.exists(folder):
    assert(False)
for _,_,files in os.walk(folder):
    for file in files:
        if file.endswith(".csv"):
            lst += [file+'\n']
        else:
            assert(False)
random.shuffle(lst)

len_data = len(lst)
print(len_data, "huray")
lst_train = []
lst_val = []
lst_test = []

train_data = int(len_data * 0.8)
val_data = int(len_data*0.1)
test_data = len_data - train_data - val_data
seg_lst = lst[0:train_data]
lst_train += seg_lst
seg_lst = lst[train_data: train_data + val_data]
lst_val += seg_lst
seg_lst = lst[train_data + val_data:]
lst_test += seg_lst

writein(lst_train, fd4)
writein(lst_test, fd5)
writein(lst_val, fd6)

fd4.close()
fd5.close()
fd6.close()

print("okay!")
