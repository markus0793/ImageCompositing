import os

path = "./"
i = 1
files = sorted(os.listdir(path))
for i in range(1,len(files),2):
    os.rename(files[i-1], f"{i:03d}_{i-1:03d}_{files[i-1][-7:]}")
    os.rename(files[i], f"{i:03d}_{i:03d}_{files[i][-7:]}")

