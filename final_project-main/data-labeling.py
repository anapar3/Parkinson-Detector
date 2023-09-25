
"""
This is the script used to combine all collected csv data files into
a single csv file.
"""

import numpy as np
import csv
import time

import labels


# print the available class label (see labels.py)
act_labels = labels.activity_labels
print(act_labels)

# specify the data files and corresponding activity label
# csv_files = ["data/walking-data/Walking-Accelerometer.csv", "data/Running_data/Running-Accelerometer.csv", "data/sitting_data/Sitting-Accelerometer.csv" , "data/plank-data/Plank-Accelerometer.csv"]

#csv_files = ["data/walking-data-NP/walking-NP.csv", "data/walking-data-YP/walking-YP.csv"]
#activity_list = ["walking no Parkinson's", "walking yes Parkinson's"]

csv_files = ["data/holding-data-NP/holding-NP.csv", "data/holding-data-YP/holding-YP.csv"]
activity_list = ["holding no Parkinson's", "holding yes Parkinson's"]
# Specify final output file name. 
output_filename = "data/All-Activities/all_labeled_data_holding.csv"

all_data = []

zip_list = zip(csv_files, activity_list)

for f_name, act in zip_list:

    if act in act_labels:
        label_id = act_labels.index(act)
    else:
        print("Label: " + act + " NOT in the activity label list! Check label.py")
        exit()
    print("Process file: " + f_name + " and assign label: " + act + " with label id: " + str(label_id))

    with open(f_name, "r") as f:
        reader = csv.reader(f, delimiter = ",")
        headings = next(reader)
        for row in reader:
            row.append(str(label_id))
            all_data.append(row)


with open(output_filename, 'w',  newline='') as f:
    writer = csv.writer(f)
    writer.writerows(all_data)
    print("Data saved to: " + output_filename)
