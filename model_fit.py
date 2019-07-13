
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
#   モデルに学習をさせる

"""
    loss はエポックが経つにつれてどんどん下がるが、
        逆に val_loss が上がっていく。
        
        これは、訓練データセットに過剰にフィットしてしまうために
            未知のデータセットに対する予測性能が下がってしまう
            過学習を起こしていることを意味する。
        
        機械学習の目的は未知のデータセットに対する予測性能を上げることなので過学習はダメ！

    普通は訓練ループを回すほど性能が上がりそうだけど、
        先に見たように訓練ループを回せば回すほど性能が悪化する場合がある。
        
        そのため、予測性能が下がる前にループを打ち切りたい。
            val_loss をプロットして目視でどこで打ち切るか判断することもできるが、
            それを自動で判断してくれるのがEarly-stoppingというアルゴリズム。

    Theanoでは自分で実装（2015/5/26）したが、
        Kerasではコールバック関数としてEarlyStoppingが実装されているため
            fit()のcallbacksオプションに設定するだけでよい(うまい実装だね～)。
        EarlyStoppingを使うには必ずバリデーションデータセットを用意する必要がある。
            fit()のオプションでvalidation_dataを直接指定することもできるが、
            validation_splitを指定することで訓練データの一部をバリデーションデータセットとして使える。

        Keras examplesもそうだが、
            テストデータセットをバリデーションデータセットとして使うのは本来ダメらしい。
            バリデーションデータセットとテストデータセットは分けたほうがよい。
            何となく直感的にダメそうというのはわかるのだがどうしてかは実はよく知らない。
"""

# -- 本題 --
from load_prep_minst import load_and_prep_minstData
from make_model import build_multilayer_perceptron
from keras.callbacks import EarlyStopping
#   https://keras.io/ja/callbacks/
from keras.optimizers import Adam

# -- load & prep --
(x_train, y_train), (x_test, y_test) = load_and_prep_minstData()

# -- build model --
model = build_multilayer_perceptron()

# -- compile model --
model.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy']
            )

# -- Early Stopping --
early_stopping = EarlyStopping(patience=0, verbose=1)

#model.summary()

# -- fit --
history = model.fit(
                        x_train, y_train,
                        batch_size=32,
                        epochs=10,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[early_stopping]
                        )

