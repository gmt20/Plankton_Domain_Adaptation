import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder as label_encoder

def createDataset(path):
    dir_list = os.listdir(path)

    images = []
    labels = []
    for dir in dir_list:
        path_dir = path + "/" + dir
        file_list = os.listdir(path_dir)
        for file in file_list:
            images.append(dir + "/" + file)
            labels.append(dir)

    # label encoder
    le = label_encoder()
    le.fit(labels)
    transformed_labels = le.transform(labels)

    # 70-30 train-test split
    image_train, image_test, label_train, label_test = train_test_split(
        images, transformed_labels, test_size=0.3, random_state=42
    )

    # 60-40 train-val split
    fin_image_train, image_val, fin_label_train, label_val = train_test_split(
        image_train, label_train, test_size=0.4, random_state=42
    )

    dataset = {
        "train_images": fin_image_train,
        "train_labels": fin_label_train,
        "val_images": image_val,
        "val_labels": label_val,
        "test_images": image_test,
        "test_labels": label_test
    }

    outfile = open("kaggle_dataset.pkl", "wb")
    pickle.dump(dataset, outfile)
    return outfile


if __name__ == "__main__":
    path = "../data/kaggle"
    createDataset(path)
