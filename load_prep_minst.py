
# Thinkpad で学習をさせることも出てきたので、
#   Keras の model.fit で行った学習の log を取得したい。

""" 現状
    使っている module
        Keras :
            model = Sequential()
                history = model.fit()
            => history 型 objett が return されるらしい。
"""

# CIFAR10 だと重いので、 MNIST を例にやってみる。

# -- import --
from keras.datasets import mnist
from keras.utils import np_utils

def load_and_prep_minstData(flatten=True, nomalize=True, one_hot=True):
    # -- load data --
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if flatten:
        # -- データ整形 --
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)

        #print("x_train.shape : ", x_train.shape)
        #print("x_test.shape : ", x_test.shape)
        #print("type(x_train) : ", type(x_train))
        #print("type(x_test) : ", type(x_test))

    if nomalize:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255.0
        x_test /= 255.0

        #print("e.g : ", x_train[0])

    if one_hot:
        class_num = 10

        y_train = np_utils.to_categorical(y_train, class_num)
        y_test = np_utils.to_categorical(y_test, class_num)

        #print('e.g : ', y_train[0])

    return (x_train, x_test), (y_train, y_test)


if __name__ == "__main__":
    # 確認
    (x_train, x_test), (y_train, y_test) = load_and_prep_minstData()

    print('confirm : ', x_train[0])
    print('confirm : ', y_train[0])
    print('confirm : ', x_test[0])
    print('confirm : ', y_test[0])

