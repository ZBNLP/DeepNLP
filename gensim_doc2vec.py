# coding:utf-8

import sys
import gensim
import sklearn
import numpy as np

from gensim.models.doc2vec import Doc2Vec, LabeledSentence

TaggededDocument = gensim.models.doc2vec.TaggedDocument

def get_datasest():
    with open('D:\zlxNLP\semantic_similarity\dataForTest\output_unique.txt', 'r',encoding='utf-8') as cf:   #分词后的文件
        docs = cf.readlines()
        # print(len(docs))

    x_train = []
    #y = np.concatenate(np.ones(len(docs)))
    for i, text in enumerate(docs):
        # print(text.replace('\n',''))
        aa=text.replace('\n','').split('\t')
        # print(len(aa))
        if len(aa)==2:
            word_list = aa[1].split('|')   #[1]表示summary
            # print('查看word_list1')
            # print(word_list)
            # l = len(word_list)
            summaryWord=[]
            for w in word_list:
               summaryWord.append(w.split(":")[0])
            summaryWord=[w.strip() for w in summaryWord]
            # l = len(summaryWord)
            # word_list[l-1] = word_list[l-1].strip()
            # print('查看word_list2')
            # print(word_list)
            document = TaggededDocument(summaryWord, tags=[i])
            # print('查看一下document是什么样的')
            # print(document)
            x_train.append(document)
    return x_train


def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1, size)) for z in corpus]
    return np.concatenate(vecs)

def train(x_train, size=200, epoch_num=1):
    model_dm = Doc2Vec(x_train,min_count=1, window = 3, vector_size = size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('D:/zlxNLP/semantic_similarity/dataForTest/model_pear')

    return model_dm

def get_datatest():
    with open('D:\zlxNLP\semantic_similarity\dataForTest\output_test.txt', 'r',encoding='utf-8') as cf:   #分词后的文件
        docs = cf.readlines()
        # print(len(docs))

    x_train = []
    #y = np.concatenate(np.ones(len(docs)))
    testDocument=[]
    for i, text in enumerate(docs):
        word_list = text.split('\t')[1].split('|')   #[1]表示summary
        l = len(word_list)
        summaryWord=[]
        for w in word_list:
           summaryWord.append(w.split(":")[0])
        # l = len(summaryWord)
        summaryWord= [w.strip() for w in summaryWord ]
        testDocument.append(summaryWord)

    return testDocument


# def test():
#
#
#     #或者是下面这个版本
#     # inferred_vector_dm = model_dm.infer_vector(testDocument)
#     # print (inferred_vector_dm)
#     # sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
#     # return sims

if __name__ == '__main__':
    x_train = get_datasest()
    # print('训练集：')
    # print(x_train)
    #模型训练
    # model_dm = train(x_train)
    model_dm = Doc2Vec.load('D:/zlxNLP/semantic_similarity/dataForTest/model_pear')
    # test_text = ['《', '舞林', '争霸' '》', '十强' '出炉', '复活', '舞者', '澳门', '踢馆']
    testDocument=get_datatest()
    print('测试集的文本量')
    print(len(testDocument))
    print(testDocument)
    for test_text in testDocument:
        print('test_text')
        print(test_text )
        inferred_vector_dm = model_dm.infer_vector(test_text)
        # print (inferred_vector_dm)
        sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)

        # print(sims)   #与各doc编号的相似度
        for count, sim in sims:
            sentence = x_train[count]
            # print(sentence)
            words = ''
            for word in sentence[0]:
                words = words + word + ' '
            print( words, sim, len(sentence[0]))

