

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
#   モデルを作る。

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
#from keras.utils.visualize_util import plot
#   https://stackoverflow.com/questions/43511819/importerror-no-module-named-keras-utils-visualize-util
# module が変わったらしい
from keras.utils.vis_utils import plot_model

def build_multilayer_perceptron():
    model = Sequential()

    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

if __name__ == "__main__":
    # built
    model = build_multilayer_perceptron()

    # summary
    model.summary()
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='plot_model.png')

    # compile
    model.compile(
                    loss='categorical_crossentropy',
                    optimizer=Adam(),
                    metrics=['accuracy']
                )

    """
        隠れ層が2つある多層パーセプトロンを構築した。
            活性化関数には relu。
        
        また、過学習を防止するテクニックである Dropout を用いた。
            Dropoutも層として追加する。

        model.summary()を使うとモデル形状のサマリが表示される。
            modelにadd()で追加した順になっている。
            
            Output ShapeのNoneはサンプル数を表している。
            dense_1 層のパラメータ数（重み行列のサイズのこと）は 784*512+512=401920 となる。
            
            512を足すのはバイアスも重みに含めているため。
            ユーザがバイアスの存在を気にする必要はないが、
            裏ではバイアスも考慮されていることがパラメータ数からわかる。
            
            同様に dense_2 層のパラメータ数は 512*512+512=262656 となる。
            同様に dense_3 層のパラメータ数は 512*10+10=5130 となる。
    """

    """
        keras.utils.vis_utils の plot_model() を使うとモデルを画像として保存できる。
            今はまだ単純なモデルなので summary() と同じでありがたみがないが
            もっと複雑なモデルだと図の方がわかりやすそう。
    """

