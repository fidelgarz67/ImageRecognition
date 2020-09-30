# Import needed modules, .json data filepath
import pandas as pd, json, os, time
t0 = time.time()

wild_json = r"C:\Users\ouahwhitenack\Desktop\wildlife\caltech_bboxes_20200316.json"
image_level_json = r"C:\Users\ouahwhitenack\Downloads\caltech_images_20200316.json\caltech_images_20200316.json"
dict_def = 'annotations' # the dictionary we want to use, also a wrapper we remove

# Load .json data in
with open(wild_json) as json_data:
    data = json.load(json_data)

# Filter .json dict down to just what we want (image annotations)
annot_dict = {k: v for k, v in data.items() if k.startswith(dict_def)}

# One more step into the tree, remove outer wrapper dictionary and df it
#inner_sanc = (annot_dict[dict_def])

df = pd.DataFrame(annot_dict[dict_def])
df = df[['category_id','image_id']] # Trim away non-essential columns
test_df = (df[:1000])

# StaticB B
sep = '/'
train_dir = r"C:\Users\ouahwhitenack\Desktop\wildlife\extract\cct_images"
step_dir = r"C:\Users\ouahwhitenack\Desktop\wildlife\extract"

print(train_dir)
animal_csv = step_dir + '\cct_animal.csv'
empty_csv = step_dir + '\cct_empty.csv'
fail_csv = step_dir + '\cct_fail.csv'

# Making file lists
animal_list = []
empty_list = []
fail_list = []

import cv2

for index, rows in df.iterrows():
    cat_id = (rows['category_id'])
    file_root = (rows['image_id'])
    file_jpg = file_root + '.jpg'
    file_full = (os.path.abspath(os.path.join(os.sep, train_dir, file_jpg)).replace('\\','/'))
    file_abs = 'r\'' + file_full + '\''
    img = cv2.imread(file_full, cv2.IMREAD_GRAYSCALE)
    if img is None:
        fail_list.append([file_full, cat_id])
    elif cat_id == 30: # cat ID 30 equals 'empty'
        #print(file_abs)
        empty_list.append([file_full, cat_id])
    else:
        animal_list.append([file_full, cat_id])

# Some result messages
total = len(animal_list) + len(empty_list) + len(fail_list)

print("There are: " + str(len(animal_list)) + " animal images. Which is: " + str(((len(animal_list)/total))*100) + " percent of total images.")
print("There are: " + str(len(empty_list)) + " empty images. Which is: " + str(((len(empty_list)/total))*100) + " percent of total images.")
print("There are: " + str(len(fail_list)) + " fail images. Which is: " + str(((len(fail_list)/total))*100) + " percent of total images.")
print("There are: " + str(len(animal_list) + len(empty_list) + len(fail_list)) + " total images.")

# Import to csv
import csv

csv_list = [fail_csv, animal_csv, empty_csv]
print(csv_list)
animal_list_list = [fail_list, animal_list, empty_list]

with open(empty_csv, 'w') as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in empty_list:
        writer.writerow([val])

with open(animal_csv, 'w') as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in animal_list:
        writer.writerow([val])

with open(fail_csv, 'w') as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in fail_list:
        writer.writerow([val])

    #filename = (rows['image_id'])
    #print(filename)

    #image = cv2.imread(src_path+rows['imagename'])
    #brand = image[rows['y']:rows['y']+rows['h'], rows['x']:rows['x']+rows['w']]
    #counter=counter+1
    #fold = rows['brand']+"/"
    #dest_fold = dest_path+fold
    #cv2.imwrite(dest_fold+"/"+filename+ "_" +str(counter)+".jpg", brand)


t1 = time.time()
total = t1-t0
print(total)
#animal_df = test_df[test_df['category_id'] != 10]
#print(empty_df)
