import sys

def get_labels(line):
    pid = int(line[3:6], base=10)
    return pid

txt = sys.argv[1]
 # read the folder then review every file here
fd = open("./table_data.txt","r")
fd2 = open("./table_{}.txt".format(txt),"w")
for file in fd:
    txt2 = "{}.csv\n".format(txt)
    if file.endswith(txt2):
        id_part = get_labels(file)
        fd2.write(file)
fd2.close()

