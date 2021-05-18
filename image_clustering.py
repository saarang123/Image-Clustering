from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
import numpy as np

model=ResNet50(weights="imagenet",include_top="False")
model.summary()

#END_OF_FIRST_CELL

import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

total_feature_list=[]

path = "clustering_dataset/"
L3=[]
for i in (os.listdir(path)):
    img_path=str(path)+str(i)
    img=image.load_img(img_path,target_size = (224,224))
    img_data=image.img_to_array(img)
    img_data=np.expand_dims(img_data,axis=0)
    img_data=preprocess_input(img_data)
    features=model.predict(img_data)
    feature_vector=np.array(features)
    total_feature_list.append(feature_vector.flatten())
    L3.append(img)
total_feature_list_np=np.array(total_feature_list)
kmeans = KMeans(n_clusters=4, random_state=0).fit(total_feature_list_np)
preds=kmeans.predict(total_feature_list_np)
pred=(preds).tolist()
for i in range(len(L3)):
    plt.imshow(L3[i])
    plt.show()
    print("The imagine is in cluster:", pred[i])
