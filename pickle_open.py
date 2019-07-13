
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
# datetime でソートした pickle file に雑に history を格納したので
#   それを参照して記録を描画してみる。

import pickle

print('参照したい history が格納されているファイルの名前を入力してください : ')
target_file = input()

with open(target_file, 'rb') as f:
    history = pickle.load(f)

print(type(history))  # <class 'keras.callbacks.History'>
print(history)  # <keras.callbacks.History object at 0xb2f744978>
#   >> <keras.callbacks.History object at 0x1008b1240>
# at 以下が変わるのだけどできるのだろうか..


# 検証がてら plot してみる。
import matplotlib.pyplot as plt


def plot_history(history):
    print(history.history.keys())

    # 認識制度(acc) の遷移を描画
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()

    # 損失関数の値の履歴を描画
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()



if __name__ == "__main__":
    plot_history(history)
    # できてるっぽいけど、epochを少なくしすぎたので、
    #   検証できない感じになってしまった..
    # print(history) したときの at 以下が毎回変わるのだけど
    #   何故なのだろう..
    #   とりあえず、history の読み書きはできるっぽいな..
    #       epoch 増やしたものを Thinkpad に計算させてみよう!
    # あと、なんで TensorFlow が起動するのかわからん..