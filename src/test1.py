import os

list1=os.listdir("/home/stgat/Downloads/pepsico")

# print(list1)
extension = []
for list in list1:

    ext = list.split(".")[-1]
    print(ext)
    if ext not in extension:
        extension.append(ext)

print(extension)