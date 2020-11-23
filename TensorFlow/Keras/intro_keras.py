import numpy as np
from tensorflow import keras


def get_dataset(training=True):
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    if training:
        return (train_images, train_labels)
    else:
        return (test_images, test_labels)


def print_stats(train_images, train_labels):
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    print(len(train_images))
    print('{}{}{}'.format(train_images[0].shape[0], 'x', train_images[0].shape[1]))
    num=np.zeros(len(class_names))
    for imgIdx in range(len(train_images)):
        num[train_labels[imgIdx]]+=1
    for i in range(len(num)):
        print('{}{} {} {} {}'.format(i,'.',class_names[i],'-',int(num[i])))

def build_model():
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="auto", name="sparse_categorical_crossentropy"),
        metrics=['accuracy']
    )
    return model

def train_model(model, train_images, train_labels, T):
    model.fit(train_images, train_labels, batch_size=32, epochs=T)

def evaluate_model(model, test_images, test_labels, show_loss=True):
    test_loss, test_accuracy=model.evaluate(test_images,test_labels,batch_size=48)
    if show_loss:
        print('{} {:.2f}'.format('Loss:',test_loss))
        print('{} {:.2f}'.format('Accuracy:', test_accuracy*100))
    else:
        print('{} {:.2f}'.format('Accuracy:', test_accuracy*100))

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
    #
    evaluate_model(model, test_images, test_labels, show_loss=True)
    # predict_label(model,test_images,1)

if __name__=="__main__":
    main()