# 基于keras的手写字体识别
# 神经网络

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


def get_dataset(training=True):
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    if training:
        return (train_images, train_labels)
    else:
        return (test_images, test_labels)


def print_stats(train_images, train_labels):
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    dic = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight',
            9: 'Nine'}
    print(len(train_images))
    print('{}{}{}'.format(train_images[0].shape[0], 'X', train_images[0].shape[1]))
    num=np.zeros(len(class_names))
    for imgIdx in range(len(train_images)):
        num[train_labels[imgIdx]]+=1
    for i in range(len(num)):
        print('{}{} {} {} {}'.format(i,'.',class_names[i],'-',num[i]))

def build_model():
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(60, activation="relu"),
        keras.layers.Dense(10,activation="softmax")
    ])
    model.compile(
        optimizer='sgd',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def train_model(model, train_images, train_labels, T):
    X_valid, X_train = train_images[:5000] / 255.0, train_images[5000:] / 255.0
    y_valid, y_train = train_labels[:5000], train_labels[5000:]
    model.fit(X_train, y_train, epochs=T,validation_data=(X_valid, y_valid))

def evaluate_model(model, test_images, test_labels, show_loss=True):
    test_loss, test_accuracy=model.evaluate(test_images,test_labels,verbose=0)
    if show_loss:
        print('{} {:.2f}'.format('Loss:',test_loss))
        print('{} {:.2f}{}'.format('Accuracy:', test_accuracy*100,'%'))
    else:
        print('{} {:.2f}{}'.format('Accuracy:', test_accuracy*100,'%'))

def predict_label(model, test_images, index):
    all_res=model.predict(test_images)
    res=all_res[index]
    idx=np.argsort(res)
    idx=idx[::-1]
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    for i in range(3):
        print('{}{} {:.2f}{}'.format(class_names[idx[i]],':', res[idx[i]]*100,'%'))



def main():
    (train_images, train_labels) = get_dataset()
    (test_images, test_labels)=get_dataset(training=False)
    # print_stats(train_images, train_labels)

    model = build_model()
    train_model(model, train_images, train_labels, 10)

    evaluate_model(model, test_images, test_labels, show_loss=True)
    # 识别test图片
    idx=int(input("输入图片索引："))
    predict_label(model,test_images,idx)
    plt.imshow(test_images[idx])
    plt.show()


if __name__=="__main__":
    main()