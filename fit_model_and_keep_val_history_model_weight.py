
# Thinkpad で学習をさせることも出てきたので、
#   Keras の model.fit で行った学習の log を取得したい。

#   => Keras の model.fit で行った学習の log を取得したい。
#       log として取るもの : 
#               history (history object -> pickle ファイル)
#               model (json 形式)
#               weight (hdf5 形式)

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
#from keras.callbacks import EarlyStopping
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
#early_stopping = EarlyStopping(patience=0, verbose=1)

#model.summary()

# -- fit --
history = model.fit(
                        x_train, y_train,
                        batch_size=32,
                        #epochs=10,
                        epochs=1,
                        verbose=1,
                        validation_split=0.1
                        #,callbacks=[early_stopping]
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


import datetime

now = datetime.datetime.now()


# 以下で取る log を保存する dir を (存在しなければ) 作成 
#   https://note.nkmk.me/python-save-file-at-new-dir/
import os
os.makedirs('outs', exist_ok=True)
outfile_pass = 'outs/out_{0:%y%m%d}_{1:%H%M%S}'.format(now, now)
os.makedirs(outfile_pass, exist_ok=True)

""" Python 3.2以降は引数 exist_ok が追加されており、
        exist_ok=Trueとすると
            既に末端ディレクトリが存在している場合もエラーが発生しない。
        末端ディレクトリが存在していなければ新規作成するし、存在していれば何もしない。
        前もって末端ディレクトリの存在確認をする必要がないので便利。
"""

""" 新しいディレクトリにファイルを作成・保存する関数のコード例
def save_file_at_new_dir(new_dir_path, new_filename, new_file_content, mode='w'):
    os.makedirs(new_dir_path, exist_ok=True)
    with open(os.path.join(new_dir_path, new_filename), mode) as f:
        f.write(new_file_content)
"""

# log として取るもの : (1)
#   history (history object -> pickle ファイル)
import pickle

#with open('history_{0:%y%m%d}_{1:%H%M%S}.pkl'.format(now, now), 'wb') as f:
#with open('./out/history_{0:%y%m%d}_{1:%H%M%S}.pkl'.format(now, now), 'wb') as f:
with open(os.path.join(outfile_pass, 'history_{0:%y%m%d}_{1:%H%M%S}.pkl'.format(now, now)), 'wb') as f:
    pickle.dump(history, f)


# log として取るもの : (2)
#   model (json 形式)
import json

model_to_json = model.to_json()
#with open('model_{0:%y%m%d}_{1:%H%M%S}.json'.format(now, now), 'w') as f:
with open(os.path.join(outfile_pass, 'model_{0:%y%m%d}_{1:%H%M%S}.json'.format(now, now)), 'w') as f:
    #json.dump(model_to_json, f)
    f.write(model_to_json)

# log として取るもの : (3)
#   weight (hdf5 形式)
#model.save_weights('weights_{0:%y%m%d}_{1:%H%M%S}.h5').format(now, now)
model.save_weights(os.path.join(outfile_pass, 'weights_{0:%y%m%d}_{1:%H%M%S}.h5').format(now, now))

# 参考 : https://keras.io/ja/models/about-keras-models/
