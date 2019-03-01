from gensim import corpora, models, similarities
import jieba
import random
import heapq
# 文本集和搜索词
# texts = ['吃鸡这里所谓的吃鸡并不是真的吃鸡，也不是我们常用的谐音词刺激的意思',
#          '而是出自策略射击游戏《绝地求生：大逃杀》里的台词',
#          '我吃鸡翅，你吃鸡腿']
# keyword = '玩过吃鸡？今晚一起吃鸡'
# # 1、将【文本集】生成【分词列表】
# texts = [jieba.lcut(text) for text in texts]
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
            x_train.append(summaryWord)
    return x_train


if __name__=='__main__':
    texts=get_datasest()
    print('查看texts')
    print(texts)
    # 2、基于文本集建立【词典】，并提取词典特征数
    dictionary = corpora.Dictionary(texts)
    feature_cnt = len(dictionary.token2id)
    print('词与编号的关系')
    print(dictionary.token2id)
    print('feature_cnt为{}'.format(feature_cnt))  #33 unique tokens
    # 3、基于词典，将【分词列表集】转换成【稀疏向量集】，称作【语料库】
    corpus = [dictionary.doc2bow(text) for text in texts]
    # 4、使用【TF-IDF模型】处理语料库
    tfidf = models.TfidfModel(corpus)
    # 5、同理，用【词典】把【搜索词】也转换为【稀疏向量】
    #从测试集中选择一个
    text10=random.sample(get_datasest(),10)
    for text in text10:
        print('选择的测试集为')
        print(text)
        kw_vector = dictionary.doc2bow(text)
        # 6、对【稀疏向量集】建立【索引】
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)
        sim = index[tfidf[kw_vector]]
        #从列表中记录最大的10个数值，及其编号，从而输出对应的文本
        nums = list(sim)
        # result = map(nums.index, heapq.nlargest(3, nums))
        temp=[]
        Inf = 0
        for i in range(10):
          temp.append(nums.index(max(nums)))
          nums[nums.index(max(nums))]=Inf
        # result.sort()
        # temp.sort()
        # print(result)
        # print(temp)
        for i in temp:
            print(texts[i],sim[i])


