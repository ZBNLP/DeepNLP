# DeepNLP
edit_consin_diff_sim.py  
将中文编辑距离，基于词频的余弦相似度（TF-IDF），以及Python difflib三种算法加权，最终的字符串比较函数compare是由0.4倍的余弦相似度，0.3倍的编辑距离相似度，0.3倍的序列化匹配加权而成。 
因为上文提到，序列化匹配和编辑距离相似度算法很相像，他们只考虑了时序关系，两者共同所占比例不应该过高。 

gensim_wmd.py
Finding similar documents with Word2Vec and WMD，具体原理参考下面链接：
http://nooverfit.com/wp/nips-2016%E8%AE%BA%E6%96%87%E7%B2%BE%E9%80%892-supervised-word-movers-distance-%E5%8F%AF%E7%9B%91%E7%9D%A3%E7%9A%84%E8%AF%8D%E7%A7%BB%E8%B7%9D%E7%A6%BB/
