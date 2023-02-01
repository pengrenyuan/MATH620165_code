from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier


max_features = 500
max_document_length = 100


def load_one_file(filename):
    x = ""
    with open(filename, encoding='gb18030', errors='ignore') as f:
        for line in f:
            line = line.strip('\n')
            line = line.strip('\r')
            x += line
    return x


def load_files_from_dir(rootdir):
    x = []
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v = load_one_file(path)
            x.append(v)
    return x


def load_all_files():
    ham = []
    spam = []
    for i in range(1, 7):
        path = "./data/enron%d/ham/" % i
        print("Load %s" % path)

        ham += load_files_from_dir(path)
        path = "./data/enron%d/spam/" % i
        print("Load %s" % path)

        spam += load_files_from_dir(path)
    return ham, spam


def get_features_by_wordbag():
    ham, spam = load_all_files()
    x = ham + spam
    y = [0] * len(ham) + [1] * len(spam)
    vectorizer = CountVectorizer(
        decode_error='ignore',
        strip_accents='ascii',
        max_features=max_features,
        stop_words='english',
        max_df=1.0,
        min_df=1)
    print(vectorizer)

    x = vectorizer.fit_transform(x)
    x = x.toarray()
    return x, y


def show_diffrent_max_features():
    global max_features
    a = []
    b = []
    for i in range(1000, 20000, 2000):
        max_features = i
        print("max_features=%d" % i)

        x, y = get_features_by_wordbag()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)
        y_pred = gnb.predict(x_test)
        score = metrics.accuracy_score(y_test, y_pred)
        a.append(max_features)
        b.append(score)
        plt.plot(a, b, 'r')
    plt.xlabel("max_features")
    plt.ylabel("metrics.accuracy_score")
    plt.title("metrics.accuracy_score VS max_features")
    plt.legend()
    plt.show()


def do_nb_wordbag(x_train, x_test, y_train, y_test):
    print("NB and wordbag")

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    # print(metrics.confusion_matrix(y_test, y_pred))



def do_svm_wordbag(x_train, x_test, y_train, y_test):
    print("SVM and wordbag")

    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))

    # print(metrics.confusion_matrix(y_test, y_pred))



def get_features_by_wordbag_tfidf():
    ham, spam = load_all_files()
    x = ham + spam
    y = [0] * len(ham) + [1] * len(spam)
    vectorizer = CountVectorizer(binary=False,
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1)
    print(vectorizer)

    x = vectorizer.fit_transform(x)
    x = x.toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    print(transformer)

    tfidf = transformer.fit_transform(x)
    x = tfidf.toarray()
    return x, y


def do_dnn_wordbag(x_train, x_test, y_train, y_testY):
    print("DNN and wordbag")


    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(5, 2),
                        random_state=1,
                        max_iter=100000)

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    # print(metrics.confusion_matrix(y_test, y_pred))



if __name__ == "__main__":
    print("--------------begin-------------- ")
    print("get_features_by_wordbag")

    # 读取数据并用词袋模型提取特征
    x, y = get_features_by_wordbag()
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    # 使用SVM，朴素贝叶斯，MLP模型训练并测试
    do_svm_wordbag(x_train, x_test, y_train, y_test)
    do_nb_wordbag(x_train, x_test, y_train, y_test)
    do_dnn_wordbag(x_train, x_test, y_train, y_test)

    print("get_features_by_wordbag_tfidf")
    # 读取数据并用TF-IDF方法提取特征
    x, y = get_features_by_wordbag_tfidf()
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    # 使用SVM，朴素贝叶斯，MLP模型训练并测试
    do_svm_wordbag(x_train, x_test, y_train, y_test)
    do_nb_wordbag(x_train, x_test, y_train, y_test)
    do_dnn_wordbag(x_train, x_test, y_train, y_test)

    # show_diffrent_max_features()

    # SVM
    # do_svm_wordbag(x_train, x_test, y_train, y_test)

    # DNN
    # do_dnn_wordbag(x_train, x_test, y_train, y_test)

    # print "get_features_by_tf"
    # x,y=get_features_by_tf()
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)
    # CNN
    # do_cnn_wordbag(x_train, x_test, y_train, y_test)

    # RNN
    # do_rnn_wordbag(x_train, x_test, y_train, y_test)