import os

folder_path = 'pics'

for i in range(1, 20):
    old_name_l = 'cornerl_{0}.jpg'.format(i)
    if old_name_l not in os.listdir(folder_path):
        continue
    else:
        new_name_l = 'left_{0}.jpg'.format(i)

    old_name_r = 'cornerr_{0}.jpg'.format(i)
    if old_name_l not in os.listdir(folder_path):
        continue
    else:
        new_name_r = 'right_{0}.jpg'.format(i)

    os.rename(os.path.join(folder_path, old_name_l), os.path.join(folder_path, new_name_l))
    os.rename(os.path.join(folder_path, old_name_r), os.path.join(folder_path, new_name_r))