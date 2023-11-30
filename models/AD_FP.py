import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
import rdkit
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
import numpy as np
import matplotlib.font_manager as font_manager
import deepchem as dc
import pandas as pd
import deepchem.feat




def get_maccs_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles) # type: ignore
    if mol is None:
        return None
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    return maccs_fp
    # 计算化合物间的谷本系数


def get_tanimoto_similarity(fp1, fp2):
    if True:
        if fp1 is None or fp2 is None:
            return 0.0
        return DataStructs.FingerprintSimilarity(fp1, fp2)


class ADFP_AC:
    def __init__(self, train, test, S, C,y_name='LogLD',min_num = 3):
        self.train = train
        self.test = test
        self.threshold = S
        self.C = C
        self.train_values = train[y_name]
        self.min_num = min_num
    # 计算MACCS指纹
    @staticmethod
    def get_maccs_fingerprint_(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        return maccs_fp

    def get_maccs_fingerprint(self,smiles_lis):
        lis = []
        for m in smiles_lis:
            fp = self.get_maccs_fingerprint_(m)
            lis.append(fp)
        return lis
    # 计算化合物间的谷本系数
    def get_tanimoto_similarity(self,fp1, fp2):
        if fp1 is None or fp2 is None:
            return 0.0
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    # 相似度矩阵计算
    def compute_S_matrix(self,smiles_lis):
        length = len(smiles_lis)
        lis = [[0]*length]*length
        
        maccs_lis = self.get_maccs_fingerprint(smiles_lis=smiles_lis)
        
        for i in tqdm(range(length)):
            for j in range(length):        
                S = self.get_tanimoto_similarity(maccs_lis[i],maccs_lis[j])
                lis[i][j] = S
                lis[j][i] = S
        return lis

    # 找到一个化合物的community
    def find_community(self,i):
        S_matrix = self.S_matrix
        lis = []
        for _x in range(len(S_matrix)):
            if S_matrix[i][_x] >= self.threshold :
                if _x != i:
                    lis.append(_x)
        return lis
    
    # 计算在其属的community中，计算其的SLD
    def cal_SLD(self,i,community_lis):
        SLD = 0
        for _x in community_lis:
            S = self.S_matrix[i][_x]
            D = abs(self.train_values[i] - self.train_values[_x])
            SLD += S*D
        return SLD/len(community_lis)
        
        
    # 定义聚类函数
    def cluster_molecules_init(self):
        '''类别初始化'''
        threshold = self.threshold
        index_lis = self.train.index
        data = self.train#.reset_index(drop=True) # 重置index？？？

        # 计算相似度矩阵
        self.S_matrix = self.compute_S_matrix(data.smiles)
        
        class_lis = [] # 指标列表：0为未分类化合物，1为ACs，2为正常化合物，
        for i in range(len(data.smiles)):
            community_lis = self.find_community(i)
            if len(community_lis) < self.min_num: # 周围化合物小于三个
                class_lis.append(0)
            else:
                SLD_i = self.cal_SLD(i,community_lis)
                if SLD_i < self.C:
                    class_lis.append(2)
                else:
                    class_lis.append(1)

        data['class'] = class_lis
        self.train = data
        print('完成训练集分类：调用self.train查看数据集')
        print(f'共找到{class_lis.count(1)}个ACs in {len(class_lis)}')
        print(f'有第一类化合物{class_lis.count(0)}种')


    def test_AD_process(self,test=None):
        if test !=None:
            self.test = test
        
        macc_i_lis = self.get_maccs_fingerprint(self.test.smiles)
        macc_j_lis = self.get_maccs_fingerprint(self.train.smiles)
        
        S_matrix_test = []
        for i in tqdm(range(len(self.test.smiles))):
            macc_i = macc_i_lis[i]
            lis = []
            for j in range(len(self.train.smiles)):
                macc_j = macc_j_lis[j]
                _S = self.get_tanimoto_similarity(macc_i,macc_j)
                lis.append(_S)
            S_matrix_test.append(lis)
        
        self.S_matrix_test = S_matrix_test
        
        class_lis =[] # 指标列表：0为未分类化合物，1为ACs，2为正常化合物，
        poi_lis = []
        for i in range(len(self.test.smiles)):
            lis = []
            for _x in range(len(S_matrix_test[i])):
                if S_matrix_test[i][_x] >= self.threshold :
                    lis.append(_x)
            
            if len(lis) < self.min_num:
                class_lis.append(0)
                poi_lis.append(1)
            else:
                # 找到lis中化合物的类型
                type = list(self.train['class'][lis])
                poi = type.count(1)/len(lis)
                poi_lis.append(poi)
                if poi > 0.5:#list(self.train['class']).count(1)/len(self.train['class']):
                    class_lis.append(1)
                else:
                    class_lis.append(2)
        # print(list(self.train['class']).count(1)/len(self.train['class']))
        self.test['class'] = class_lis
        self.test['poi'] = poi_lis
        print(f'测试集处理结束，适用域外的化合物共有{class_lis.count(0)+class_lis.count(1)}个')