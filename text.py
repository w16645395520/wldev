from bert_serving.client import BertClient
import time
import numpy as np
tx1 = '姓名'
tx2 = '骨折手术'
t1 = time.time()
bc = BertClient()
a = bc.encode([tx1, tx2])
print('time:',time.time()-t1)

adict = dict()
adict[tx1] = a[0]
adict[tx2] = a[1]

sum = 0
for i,j in zip(adict[tx1],adict[tx2]):
    sum += (j-i)*2
print(sum**2)


'''
bert-serving-start -model_dir /home/wanglei/algorithm/github/bert-bilstm-crf-ner/bert_base/chinese_L-12_H-768_A-12 -num_worker=4 

bert-serving-start -model_dir /home/wanglei/algorithm/github/bert-bilstm-crf-ner/bert_base/chinese_L-12_H-768_A-12 -num_worker=1 

'''
'''
pip install bert-serving-server
pip install bert-serving-client

新增住院申请单、挂号单、门诊病历种类提取
病案首页新增点位
'''
