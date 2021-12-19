classes_array = []
# read file
f = open("test.txt", encoding="utf-8")
line = f.readline()
i = 1
while line:
    line = f.readline()
    if not line:
        break
    sentence_front_part = line.split("\t")[0]
    classes = sentence_front_part.split(",")
    sentence = line.split("\t")[1]
    for single_class in classes:
        if single_class not in classes_array:
            classes_array.append(single_class)
f.close()
print(classes_array)

f = open('class.txt', 'w', encoding="utf-8")
for single_class in classes_array:
    f.write(single_class+"\n")
