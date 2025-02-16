import csv
import cv2
import os
import pandas as pd
if not os.path.exists('./dataset'):
    os.makedirs('./dataset')

name = input("enter your name")
roll = input("enter your id")
row = [name,roll,'A']
l = []
for root ,dire,filenames in os.walk('dataset'):
    
    for names in dire:
        print(names)
        l.append(int(names))
    
if(len(l) == 0):
    folder = 0
else:
    folder = str(l[-1]+1)

os.makedirs(f'./dataset/{folder}')

'''def add(row):
    with open('data.csv','a') as f:
        writer = csv.writer(f,lineterminator='\n')
        writer.writerow(row)'''
def add_to_csv(row):
    try:
        # Open the CSV file in append mode, creating it if it doesn't exist
        with open('data.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            print('Data added to data.csv successfully!')
    except Exception as e:
        print(f"Error adding data to data.csv: {e}")

add_to_csv(row)        
df = pd.read_csv('data.csv')  # Load CSV into a DataFrame
df.to_excel('data.xlsx', index=False)  # Save as Excel
print("Data also saved to 'data.xlsx'.")
   
# with open('data.csv') as f:
#     data = csv.reader(f)
#     next(data)
#     for names in data:
#         if names[0] == name:
#             print('already exist!!')
#             break
#         else:
#             add(row)
#             print('added')
#             break
#         print(names)

capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
framecount = 0 

flag,image = capture.read()

while True:
    flag,frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        framecount += 1
        cv2.imwrite(f'dataset/{folder}/{name}.{roll}.{framecount}.jpg',frame)
        print('frame no',framecount,' captured!')
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.waitKey(100)
    cv2.imshow('img',frame)
    cv2.waitKey(1)
    if framecount >= 1000:
        break


capture.release()

cv2.destroyAllWindows()





 