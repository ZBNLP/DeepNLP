import jieba
import numpy as np
import difflib
from collections import Counter

#其中的str1，str2是分词后的标签列表
def edit_similar(str1,str2):
    len_str1=len(str1)
    len_str2=len(str2)
    taglist=np.zeros((len_str1+1,len_str2+1))
    for a in range(len_str1):
        taglist[a][0]=a
    for a in range(len_str2):
        taglist[0][a] = a
    for i in range(1,len_str1+1):
        for j in range(1,len_str2+1):
            if(str1[i - 1] == str2[j - 1]):
                temp = 0
            else:
                temp = 1
            taglist[i][j] = min(taglist[i - 1][j - 1] + temp, taglist[i][j - 1] + 1, taglist[i - 1][j] + 1)
    return 1-taglist[len_str1][len_str2] / max(len_str1, len_str2)


#其中的str1，str2是分词后的标签列表
def cos_sim(str1, str2):
    co_str1 = (Counter(str1))
    co_str2 = (Counter(str2))
    p_str1 = []
    p_str2 = []
    for temp in set(str1 + str2):
        p_str1.append(co_str1[temp])
        p_str2.append(co_str2[temp])
    p_str1 = np.array(p_str1)
    p_str2 = np.array(p_str2)
    return p_str1.dot(p_str2) / (np.sqrt(p_str1.dot(p_str1)) * np.sqrt(p_str2.dot(p_str2)))





def compare(str1, str2):
    if str1 == str2:
        return 1.0
    #其中的str1，str2并未分词，是两组字符串
    diff_result=difflib.SequenceMatcher(None,str1,str2).ratio()
    print('diff相似度')
    print(diff_result)
    # #分词
    # str1=jieba.lcut(str1)
    # str2 = jieba.lcut(str2)
    cos_result=cos_sim(str1, str2)
    print('余弦相似度')
    print(cos_result)
    edit_result=edit_similar(str1,str2)
    print('编辑距离')
    print(edit_result)
    return cos_result*0.4+edit_result*0.3+0.3*diff_result

if __name__ == "__main__":
    str1=['我','爱','漫威']
    str2=['我','喜欢','漫威','电影']
    sim=compare(str1, str2)
    print('加权后的相似度')
    print(sim)


