
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
#   モデルが学習した結果(変数) を保持したままにしたい。(プログラム外から参照できるようにする。)

from load_prep_minst import load_and_prep_minstData
from build_model import build_multilayer_perceptron
from keras.callbacks import EarlyStopping
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
                        #epochs=10,
                        epochs=3,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[early_stopping]
                        )


#print(type(history))  # <class 'keras.callbacks.History'>
#print(history)  # <keras.callbacks.History object at 0xb36762f98>

""" fit() method の戻り値:
    History オブジェクト． 
        History.history 属性は
            * 実行に成功したエポックにおける訓練の損失値と
            * 評価関数値の記録と，
            * (適用可能ならば) 検証における損失値と評価関数値
        も記録しています．
"""

# https://qiita.com/karadaharu/items/948e4d313fbaa32e408c

import pickle, datetime

now = datetime.datetime.now()

with open('history_{0:%y%m%d}_{1:%H%M%S}.pkl'.format(now, now), 'wb') as f:
    pickle.dump(history, f)