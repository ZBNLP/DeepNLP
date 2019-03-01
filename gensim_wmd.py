from time import time
start_nb = time()
from jpype import *
import json
startJVM(getDefaultJVMPath(), "-Djava.class.path=D:\zlxNLP\hanlp-1.7.1.jar;D:\zlxNLP", "-Xms1g", "-Xmx1g") # 启动JVM，Linux需替换分号;为冒号:
# print("="*30+"HanLP分词"+"="*30)
HanLP = JClass('com.hankcs.hanlp.HanLP')

# Initialize logging.
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

start = time()
import os

from gensim.models import KeyedVectors
if not os.path.exists('D:\zlxNLP\semantic_similarity\dataForTest\sgns.sogou.txt'):
    raise ValueError("SKIP: You need to download the google news model")

model = KeyedVectors.load_word2vec_format('D:\zlxNLP\semantic_similarity\dataForTest\sgns.sogou.txt', binary=False)

print('Cell took %.2f seconds to run.' % (time() - start))


sentence_obama=['我','爱','漫威']
sentence_president=['我','喜欢','漫威','电影']
distance = model.wmdistance(sentence_obama, sentence_president)
print('distance = %.4f' % distance)
sentence_orange  =['我','是','漫威迷']
distance = model.wmdistance(sentence_obama, sentence_orange)
print('distance = %.4f' % distance)
# Normalizing word2vec vectors.
start = time()
model.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.
distance = model.wmdistance(sentence_obama, sentence_president)  # Compute WMD as normal.
print('distance = %.4f' % distance)
distance = model.wmdistance(sentence_obama, sentence_orange)
print('distance = %.4f' % distance)

print('Cell took %.2f seconds to run.' %(time() - start))

if __name__ == '__main__':
    #第一步分词
    # segmentFile=open("D:/zlxNLP/semantic_similarity/dataForTest/pear_summary_segment.txt",'w', encoding='UTF-8')
    # for line in open("D:/zlxNLP/semantic_similarity/dataForTest/pear_summary.txt",'r', encoding='UTF-8'):
    #     content=line.replace('\n','').split('\t')
    #     #运用hanlp分词，#运用wmd计算summary的相似度
    #     if len(content)==2:
    #         # 中文分词
    #         summary_term_list= HanLP.segment(content[1])   #分词不要词性,分词后需要过滤掉标点符号
    #         # print([str(i.word) for i in summary_term_list and  if  i.nature.find('w')==-1 ])
    #         summaryWords= '|'.join([str(i.word) for i in summary_term_list])
    #         print(summaryWords)
    #         # segmentFile.write(summaryWords+'\n')
    '''
    #训练过程
    #分词结束，读取全部内容
    logfile=open("D:/zlxNLP/semantic_similarity/dataForTest/summary_sim.txt",'w', encoding='UTF-8')
    logfile1=open("D:/zlxNLP/semantic_similarity/dataForTest/summary_sim_1.txt",'w', encoding='UTF-8')
    simDict={}
    i=0
    for line in open("D:/zlxNLP/semantic_similarity/dataForTest/output.txt",'r', encoding='UTF-8'):

           if i <2:
               i+=1
               print(i)
               content=line.replace('\n','').split('\t')
               summaryWords1=[]
               if len(content)==2:
                   term_list=content[1].split('|')
                   for w in term_list:
                       summaryWords1.append(w.split(":")[0])
               # print(summaryWords1)
               for line in open("D:/zlxNLP/semantic_similarity/dataForTest/output.txt",'r', encoding='UTF-8'):
                   content=line.replace('\n','').split('\t')
                   summaryWords2=[]
                   if len(content)==2:
                       term_list=content[1].split('|')
                       for w in term_list:
                           summaryWords2.append(w.split(":")[0])
                   # print(summaryWords2)
                   distance = model.wmdistance(summaryWords1, summaryWords2)
                   simDict[('|'.join(summaryWords1)+'  和   '+ '|'.join(summaryWords2))]=distance
                   logfile1.write('{}   和   {}distance ={:.2f}' .format ('|'.join(summaryWords1), '|'.join(summaryWords2),distance)+'\n')
    #对simDict进行排序   distance越小，则两个summary越相似
    # after=dict(sorted(simDict.items(),key = lambda x:x[1],reverse =True))  #降序排列

    after=dict(sorted(simDict.items(),key = lambda x:x[1],reverse =False))  #升序排列

    # logfile.write(str(after))
    # logfile.close()

    js = json.dumps(after)
    # file = open('test.txt', 'w')
    logfile.write(js)
    logfile.close()
    # 取出前几个， 也可以在sorted返回的list中取前几个
    cnt = 0
    for key, value in after.items():
        # print("{}:{}".format(key, value))
        if value>0:
            cnt += 1
            if cnt > 100:
                break
            print("{}:{}".format(key, value))
    '''
    #读取字典
    file= open("D:/zlxNLP/semantic_similarity/dataForTest/summary_sim.txt",'r',encoding='utf-8')
    js = file.read()
    dic = json.loads(js)
    file.close()

    # 取出前几个， 也可以在sorted返回的list中取前几个
    cnt = 0
    for key, value in dic.items():
        if value>0:
            cnt += 1
            if cnt > 100:
                break
            print("{}:{}".format(key, value))













