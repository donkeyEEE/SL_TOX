# 修改813 准备将单个模型从集成模型类中分离出来，让框架更加稳定，并且可以支持多次运行单种模型
import warnings
import numpy as np




warnings.filterwarnings("ignore")

from deepchem.models.torch_models import GATModel

import chemprop

import deepchem as dc
import deepchem.data
import deepchem.models

import torch

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy.optimize import nnls
import pandas as pd

from sklearn import svm

import models.utils 

# import plot_parity, run_fun_SVR, run_fun_RF, run_fun_AFP_GAT,run_fun_DNN, dataloader_RF_SVR, dataloader_PytorchModel, ADFP_AC



# from deepchem.models.torch_models import AttentiveFPModel, MATModel, GATModel

# ===========

def dataloader_AFP(data_set :pd.DataFrame):
    """AFP模型的数据加载器，以及剔除不可使用的数据

    Args:
        data_set (pd.DataFrame): 数据集

    Returns:
        data_set: 返回numpydataset用于模型训练，其中以及剔除不可使用的数据
        empty_lis: 空元素在原df中的行索引
    """
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True) # type: ignore
    dataset, empty_lis = models.utils.dataloader_PytorchModel_629(data_set, featurizer)
    return dataset,empty_lis

def dataloader_GAT_GCN(data_set :pd.DataFrame):
    """GAT模型的数据加载器，以及剔除不可使用的数据

    Args:
        data_set (pd.DataFrame): 数据集

    Returns:
        data_set: 返回numpydataset用于模型训练，其中以及剔除不可使用的数据
        empty_lis: 空元素在原数据中的位置，
    """
    featurizer = dc.feat.MolGraphConvFeaturizer()  # type: ignore
    dataset, empty_lis = models.utils.dataloader_PytorchModel_629(data_set, featurizer)
    return dataset,empty_lis

def dataloader_DNN_RF_SVR(data_set:pd.DataFrame , ECFP_Params=[4096, 2]):
    data_set = models.utils.dataloader_RF_SVR(data_set, ECFP_Params)
    return data_set

class meta_model:
    def __init__(self,id=0,save=False,model_save_files= "Stacking_model\\model_checkpoint") -> None:
        self.save  = save
        self.id = id
        self.model_save_files = model_save_files
        self.mark = None # 用于最后显示模型结构的标识
        self.name = ""
    
    def get_model_mark(self):
        return f"id={self.id},{self.mark}"
    
class meta_AFP(meta_model):
    def __init__(self,id=0,save=False,model_save_files= "Stacking_model\\model_checkpoint") -> None: # type: ignore
        """_summary_
        Args:
            train_set (dc.data.NumpyDataset): 转换过的训练集
            test_set (dc.data.NumpyDataset): 转换过的测试集
            id (int, optional):若有多个AFP模型则需要传入不同的id以区分
            save: 是否保存模型
            model_save_files: 模型储存位置 "Stacking_model\\model_checkpoint"
        """
        super(meta_AFP,self).__init__(id,save,model_save_files)
        self.mark = "deepchem.models.AttentiveFPModel:Model for Graph Property Prediction"
        self.name = "meta_AFP"
    def get_AFP(self,train_set :dc.data.NumpyDataset,test_set: dc.data.NumpyDataset,): # type: ignore 
        """AFP模型的运行函数deepchem.models.AttentiveFPModel:Model for Graph Property Prediction.
        Return:
            recorder_AFP(dic): 训练和预测结果
                {'train_true': y_train, 'train_pre': train_pre, 'test_true': y_val, 'test_pre': pre}
            model_AFP :训练结束的模型
        """
        print('start training AFP ')
        
        
        model_AFP = dc.models.AttentiveFPModel(mode='regression', n_tasks=1,batch_size=32, learning_rate=0.001) # type: ignore
        
        
        recorder_AFP, model_AFP= models.utils.run_fun_AFP_GAT(model_AFP, 
                                                    train_dataset=train_set, test_dataset=test_set, epoch=40)
        if self.save:
            model_AFP.save_checkpoint(model_dir='{}/AFP_{}'.format(self.model_save_files,self.id))
        return recorder_AFP, model_AFP
    
    def load_predict(self,data:dc.data.NumpyDataset): # type: ignore
        print('AFP predict')
        model_AFP = dc.models.AttentiveFPModel(n_tasks=1, # type: ignore
                                            batch_size=16, learning_rate=0.001,
                                            )
        model_AFP.restore(model_dir='{}/AFP_{}'.format(self.model_save_files,self.id))
                
        pre_AFP = model_AFP.predict(data)
        return np.array(pre_AFP).reshape(-1) # type: ignore
    

class meta_GAT(meta_model):
    def __init__(self,id=0,save=False,model_save_files= "Stacking_model\\model_checkpoint") -> None: # type: ignore
        super(meta_GAT,self).__init__(id,save,model_save_files)
        self.mark = "Model for Graph Property Prediction Based on Graph Attention Networks (GAT)."
        self.name = "meta_GAT"
    def get_GAT(self,train_set :dc.data.NumpyDataset,test_set: dc.data.NumpyDataset,): # type: ignore 
        """Model for Graph Property Prediction Based on Graph Attention Networks (GAT).
        Args:
            train_set (dc.data.NumpyDataset): 转换过的训练集
            test_set (dc.data.NumpyDataset): 转换过的测试集
            id (int, optional):若有多个模型则需要传入不同的id以区分
            save: 是否保存模型
            model_save_files: 模型储存位置 "Stacking_model\\model_checkpoint"
        Return:
            recorder_GAT(pd.DataFrame): 训练和预测结果
                {'train_true': y_train, 'train_pre': train_pre, 'test_true': y_val, 'test_pre': pre}
            model_GAT :训练结束的模型
        """
        print('start training GAT ')
        
        model_GAT = GATModel(mode='regression', n_tasks=1,batch_size=32, learning_rate=0.001)
        recorder_GAT, model_GAT= models.utils.run_fun_AFP_GAT(model_GAT,
                                                            train_dataset=train_set, test_dataset=test_set, epoch=40)
        if self.save:
            model_GAT.save_checkpoint(model_dir='{}/GAT_{}'.format(self.model_save_files,self.id))
        return recorder_GAT,model_GAT
    def load_predict(self,data:dc.data.NumpyDataset): # type: ignore
        print('can not use GAT to predict')
        pass



class meta_GCN(meta_model):
    def __init__(self, id=0, save=False, model_save_files="Stacking_model\\model_checkpoint") -> None:
        super().__init__(id, save, model_save_files) 
        self.mark = "Model for Graph Property Prediction Based on Graph Convolution Networks (GCN)."
        self.name = "meta_GCN"
    def get_GCN(self, train_set: dc.data.NumpyDataset, test_set: dc.data.NumpyDataset): # type: ignore 
        """Model for Graph Property Prediction Based on Graph Convolution Networks (GCN).

        Args:
            train_set (dc.data.NumpyDataset): 转换过的训练集
            test_set (dc.data.NumpyDataset): 转换过的测试集
            id (int, optional):若有多个模型则需要传入不同的id以区分
            save: 是否保存模型
            model_save_files: 模型储存位置 "Stacking_model\\model_checkpoint"
        Return:
            recorder_GCN(pd.DataFrame): 训练和预测结果
                {'train_true': y_train, 'train_pre': train_pre, 'test_true': y_val, 'test_pre': pre}
            model_GCN :训练结束的模型
        """
        print('start training GCN ')
        from deepchem.models.torch_models import GCNModel
        model_GCN = GCNModel(mode='regression', n_tasks=1,
                                            batch_size=32, learning_rate=0.001
                                            )
        recorder_GCN, model_GCN= models.utils.run_fun_AFP_GAT(model_GCN,
                                                            train_dataset=train_set, test_dataset=test_set, epoch=40)
        if self.save:
            model_GCN.save_checkpoint(model_dir='{}/GCN_{}'.format(self.model_save_files,self.id))
        return recorder_GCN, model_GCN
    def load_predict(self,data:dc.data.NumpyDataset): # type: ignore
        print('can not use GAT to predict')
        pass
    
class meta_MPNN:
    def __init__(self, id=0, save=False, model_save_files="Stacking_model\\model_checkpoint") -> None:
        self.save  = save
        self.id = id
        self.model_save_files = model_save_files
        self.mark = None # 用于最后显示模型结构的标识
        self.mark = "Message passing neural network for molecular property prediction."
        self.name = "meta_MPNN"
    def get_model_mark(self):
        return f"id={self.id},{self.mark}"
    
    def get_MPNN(self,train_set:pd.DataFrame, test_set:pd.DataFrame,):
        print('start training MPNN ')
        ## 将数据按照chemprop读取格式进行储存
        train_set[['smiles', 'LogLD']].to_csv('{}/MPNN/train.csv'.format(self.model_save_files), index=False)
        test_set[['smiles', 'LogLD']].to_csv('{}/MPNN/test.csv'.format(self.model_save_files), index=False)
        
        ## 开始训练
        recorder_MPNN = {'train_true': np.array(train_set.LogLD), 'test_true': np.array(test_set.LogLD)}
        arguments = [
                    '--data_path', '{}/MPNN/train.csv'.format(self.model_save_files),
                    '--separate_test_path', '{}/MPNN/test.csv'.format(self.model_save_files),
                    '--separate_val_path', '{}/MPNN/test.csv'.format(self.model_save_files),
                    '--dataset_type', 'regression',
                    '--save_dir', '{}/MPNN/test_checkpoints_reg'.format(self.model_save_files),
                    '--epochs', '25',  #
                    '--num_folds', '1',
                    '--num_worker','1',
                    '--ffn_num_layers', '2',
        ]

        args = chemprop.args.TrainArgs().parse_args(arguments) # type: ignore
        mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training) # type: ignore
                
        ## 如果要连续预测多组分子，更高效的做法是只加载chemprop模型一次，然后使用预加载的模型进行预测。
                
        """
                arguments = [
                    '--test_path', '{}/MPNN/test.csv'.format(model_save_files),
                    '--preds_path', '{}/MPNN/test1.csv'.format(model_save_files),
                    '--checkpoint_dir', '{}/MPNN/test_checkpoints_reg'.format(model_save_files),
                ]

                args = chemprop.args.PredictArgs().parse_args(arguments)
                
                """
        
        arguments = [
            '--test_path', '{}/MPNN/test.csv'.format(self.model_save_files),
            '--preds_path', '/dev/null',
            '--checkpoint_dir', '{}/MPNN/test_checkpoints_reg'.format(self.model_save_files)
        ]
                
        args = chemprop.args.PredictArgs().parse_args(arguments) # type: ignore
        model_objects = chemprop.train.load_model(args=args)  # type: ignore
                
        def series_to_2Dlis(serise):
            lis = []
            for x in serise:
                lis.append([x])
            return lis
        
        def lis_to_array(lis):
            lisss= [ x[0] for x in lis]
            return np.array(lisss)
            
            
        # print(recorder_MPNN)
        test_smiles = series_to_2Dlis(test_set.smiles)
        train_smiles = series_to_2Dlis(train_set.smiles)
        test_preds = chemprop.train.make_predictions(args=args, smiles = test_smiles ,model_objects=model_objects )  # type: ignore        
        recorder_MPNN['test_pre'] = lis_to_array(test_preds)
        # print(recorder_MPNN['test_pre'])
        
        preds_train = chemprop.train.make_predictions(args=args, smiles = train_smiles ,model_objects=model_objects ) # type: ignore
        recorder_MPNN['train_pre'] = lis_to_array(preds_train)
        # print(type(recorder_MPNN))
        # print("===========\n",recorder_MPNN)
        
        print('============MPNN over============')
        
        return recorder_MPNN
    def load_predict(self,predict_data:pd.DataFrame):
        print('MPNN predict')
        predict_data[['smiles', 'LogLD']].to_csv('{}/MPNN/predict_df.csv'.format(self.model_save_files), index=False)
        arguments = [
                '--test_path', '{}/MPNN/predict_df.csv'.format(self.model_save_files),
                '--preds_path', '{}/MPNN/predict_df2.csv'.format(self.model_save_files),
                '--checkpoint_dir', '{}/MPNN/test_checkpoints_reg'.format(self.model_save_files),
            ]
        args = chemprop.args.PredictArgs().parse_args(arguments) # type: ignore
        pred_MPNN = chemprop.train.make_predictions(args=args) # type: ignore
        return np.array(pred_MPNN).reshape(-1)



"""
class meta_DNN_10_24(meta_model):
    <10.24>
    修改为以tensorflow为主体的模型。因为使用deepchem和torch，拟合效果不好。虽然使用的是同一个结构。
    可能出了些错误，暂时修改为tensor模型。

    Args:
        meta_model (_type_): _description_
    
    def __init__(self, id=0, save=False,ECFP_Params=[4096, 2] ,model_save_files="Stacking_model\\model_checkpoint",num_layers=3) -> None:
        super().__init__(id, save, model_save_files)
        self.save  = save
        self.mark = "Deep neural network useing ECFP"
        self.id = id
        self.ECFP_Params = ECFP_Params
        self.name = "meta_DNN"
        self.num_layers=num_layers
        
    def get_model_mark(self):
        return f"id={self.id},{self.mark} {self.ECFP_Params}"
    
    def get_DNN(self,train_set: dc.data.NumpyDataset, test_set: dc.data.NumpyDataset): # type: ignore
        # 定义模型
        class DNN(keras.Model):
            _n_layers = 1
            _layer_size = 16
            batch_size = 32
            learning_rate = 0.0001
            epochs = 500
            seed = 9700

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

                self.generate_fcn()

            def generate_fcn(self):
                self.pipeline = []

                for i, layer in enumerate(range(self.n_layers)):
                    self.pipeline.append(layers.BatchNormalization())
                    self.pipeline.append(layers.Dense(self.layer_size, activation='relu'))
                
                self.pipeline.append(layers.BatchNormalization())
                self.pipeline.append(layers.Dense(1, activation='linear'))


            @property
            def n_layers(self):
                return self._n_layers

            @n_layers.setter
            def n_layers(self, value):
                self._n_layers = value
                self.generate_fcn()

            @property
            def layer_size(self):
                return self._layer_size

            @layer_size.setter
            def layer_size(self, value):
                self._layer_size = value
                self.generate_fcn()

            def call(self, inputs):
                x = inputs

                for layer in self.pipeline:
                    x = layer(x)
                    
                return x

            def fit(self, x_train, y_train, **kwargs):
                tf.random.set_seed(self.seed)

                adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

                super().build(input_shape=x_train.shape)
                super().compile(optimizer=adam, loss='mse', metrics=['mse', 'mae'])
                super().fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, **kwargs)

        model = DNN()
        model.learning_rate = 0.001
        model.n_layers = 3
        model.layer_size = self.ECFP_Params[0]
        model.batch_size = 126
        model.epochs = 50
        
        DNN_model = deepchem.models.KerasModel(model, 
                                            loss=dc.models.losses.L2Loss(),
                                            optimizer=deepchem.models.optimizers.Adam(learning_rate=0.001),
                                            # batch_size=512,
                                            model_dir='{}/DNN_{}'.format(self.model_save_files,self.id)
                                            ) # type: ignore
        
        recorder_DNN, model_DNN = models.utils.run_fun_DNN(DNN_model, 
                                                        train_dataset=train_set, test_dataset=test_set,
                                                        ECFP_Params=self.ECFP_Params)
        if self.save:
            model_DNN.save_checkpoint(max_checkpoints_to_keep =1) 
        return recorder_DNN, model_DNN
    def load_predict(self,predict_data):
        print('DNN predict')
        DNN_model = torch.nn.Sequential(
                    torch.nn.Linear(self.ECFP_Params[0], self.ECFP_Params[0]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.ECFP_Params[0], self.ECFP_Params[0]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.ECFP_Params[0], 1)
                )
        DNN_model = deepchem.models.TorchModel(DNN_model, loss=dc.models.losses.L2Loss(),model_dir='{}/DNN_{}'.format(self.model_save_files,self.id))# type: ignore
        DNN_model.restore()
        pre_DNN = DNN_model.predict(predict_data)
        return np.array(pre_DNN).reshape(-1)
"""
class meta_DNN(meta_model):
    def __init__(self, id=0, save=False,ECFP_Params=[4096, 2] ,model_save_files="Stacking_model\\model_checkpoint",num_layers=3) -> None:
        super().__init__(id, save, model_save_files)
        self.mark = "Deep neural network useing ECFP"
        self.ECFP_Params = ECFP_Params
        self.name = "meta_DNN"
        self.num_layers=num_layers
        
    def get_model_mark(self):
        return f"id={self.id},{self.mark} {self.ECFP_Params}"
    
    def get_DNN(self,train_set: dc.data.NumpyDataset, test_set: dc.data.NumpyDataset): # type: ignore
        # 定义模型
        layer_lis = []
        for i in range(self.num_layers):
            layer_lis.append(torch.nn.BatchNorm1d(num_features=self.ECFP_Params[0]))
            layer_lis.append(torch.nn.Linear(self.ECFP_Params[0], self.ECFP_Params[0]))
            layer_lis.append(torch.nn.ReLU())
        
        layer_lis.append(torch.nn.Linear(int(self.ECFP_Params[0]), 1))
        
        DNN_model = torch.nn.Sequential(
            *layer_lis
        )
        optimizer = deepchem.models.optimizers.Adam()
        DNN_model = deepchem.models.TorchModel(DNN_model, 
                                            loss=dc.models.losses.L2Loss(),
                                            optimizer=optimizer,
                                            # batch_size=512,
                                            model_dir='{}/DNN_{}'.format(self.model_save_files,self.id)
                                            ) # type: ignore
        
        recorder_DNN, model_DNN = models.utils.run_fun_DNN(DNN_model, 
                                                        train_dataset=train_set, test_dataset=test_set,
                                                        ECFP_Params=self.ECFP_Params)
        if self.save:
            model_DNN.save_checkpoint(max_checkpoints_to_keep =1) 
        return recorder_DNN, model_DNN
    
    def load_predict(self,predict_data):
        print('DNN predict')
        DNN_model = torch.nn.Sequential(
                    torch.nn.Linear(self.ECFP_Params[0], self.ECFP_Params[0]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.ECFP_Params[0], self.ECFP_Params[0]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.ECFP_Params[0], 1)
                )
        DNN_model = deepchem.models.TorchModel(DNN_model, loss=dc.models.losses.L2Loss(),model_dir='{}/DNN_{}'.format(self.model_save_files,self.id))# type: ignore
        DNN_model.restore()
        pre_DNN = DNN_model.predict(predict_data)
        return np.array(pre_DNN).reshape(-1)

class meta_GBM(meta_model):
    """用于补充的模型,决策树数量为 200，学习率值为 1.2 的梯度提升树模型

    Args:
        meta_model (_type_): _description_
    """
    def __init__(self, id=0, ECFP_Params=[4096, 2],save=False, model_save_files="Stacking_model\\model_checkpoint") -> None:
        super().__init__(id, save, model_save_files)
        self.mark = "Gradient Boosting Machine useing ECFP"
        self.ECFP_Params = ECFP_Params
        self.name = "meta_GB,"
    def get_GBM(self,train_set: deepchem.data.NumpyDataset, test_set: deepchem.data.NumpyDataset):  # type: ignore 
        print('start training GBM ')
        import sklearn.ensemble 
        
        model = sklearn.ensemble.GradientBoostingRegressor(learning_rate=1.2 , n_estimators=200)
        
        model = deepchem.models.SklearnModel(model,
                        model_dir='{}/GBM_{}'.format(self.model_save_files,self.id))
        recorder, model = models.utils.run_fun_RF(model, train_dataset=train_set, test_dataset=test_set,
                                        ECFP_Params=self.ECFP_Params)
        if self.save:
            model.save()
        return recorder, model
    
class meta_RF(meta_model):
    def __init__(self, id=0, save=False,
                ECFP_Params=[4096, 2] ,
                n_estimators=181,
                min_samples_split=14,
                model_save_files="Stacking_model\\model_checkpoint") -> None:
        super().__init__(id, save, model_save_files)
        self.mark = "Random Forest Regressor useing ECFP"
        self.ECFP_Params = ECFP_Params
        self.name = "meta_RF"
        self.n_estimators = n_estimators
        self.min_samples_split  = min_samples_split
    def get_model_mark(self):
        return f"id={self.id},{self.mark} {self.ECFP_Params}"
    
    def get_RF(self,train_set: dc.data.NumpyDataset, test_set: dc.data.NumpyDataset): # type: ignore
        # print('start training RF ')
        model_RF = deepchem.models.SklearnModel(RandomForestRegressor(n_estimators=self.n_estimators, min_samples_split=self.min_samples_split),
                        model_dir='{}/RF_{}'.format(self.model_save_files,self.id))
        
        recorder_RF, model_RF = models.utils.run_fun_RF(model_RF, train_dataset=train_set, test_dataset=test_set,
                                        ECFP_Params=self.ECFP_Params)
        if self.save:
            model_RF.save()
        return recorder_RF, model_RF
    def load_predict(self,data):# type: ignore
        # print('RF predict')
        model_RF = deepchem.models.SklearnModel(RandomForestRegressor(n_estimators=181, min_samples_split=14),
                                                model_dir='{}/RF_{}'.format(self.model_save_files,self.id))
        print(self.id)
        model_RF.reload()
        pre_RF = model_RF.predict(data)        
        return np.array(pre_RF).reshape(-1) # type: ignore

class meta_SVR(meta_model):
    def __init__(self, 
                id=0, 
                save=False,
                ECFP_Params=[4096, 2],
                C=1,
                model_save_files="Stacking_model\\model_checkpoint") -> None:
        super().__init__(id, save, model_save_files)
        self.ECFP_Params = ECFP_Params
        self.name = "meta_SVR"
        self.C = C
        self.mark = f"Support vector regression useing ECFP "
    def get_model_mark(self):
        return f"id={self.id},{self.mark} {self.ECFP_Params}"
    def get_SVR(self, train_set: dc.data.NumpyDataset, test_set: dc.data.NumpyDataset): # type: ignore
        print('start training SVR ')
        model_SVR = dc.models.SklearnModel(svm.SVR(C=self.C), # type: ignore
                                                model_dir='{}/SVR_{}'.format(self.model_save_files,self.id)) # type: ignore
        recorder_SVR, model_SVR = models.utils.run_fun_RF(model_SVR, train_dataset=train_set, test_dataset=test_set,
                                                        ECFP_Params=self.ECFP_Params)
        if self.save:
            model_SVR.save()
        return recorder_SVR, model_SVR
    def load_predict(self,predict_data):
        print('SVR predict')
        model_SVR = deepchem.models.SklearnModel(svm.SVR(C=1),
                                            model_dir='{}/SVR_{}'.format(self.model_save_files,self.id))
        model_SVR.reload()
        pre_SVR = model_SVR.predict(predict_data)
        return np.array(pre_SVR).reshape(-1)

class L2_RF(meta_model):
    def __init__(self, id=0, save=False, model_save_files="Stacking_model\\model_checkpoint") -> None:
        super().__init__(id, save, model_save_files)
        self.mark = "Random Forest Regressor for the second layer of Stacking_model"
        self.name = "L2_RF"
    
    def get_model(self,recorder: dict):
        print('Start fitting RF in L2')
        model = RandomForestRegressor()
        if 'train_data' in recorder and 'train_true' in recorder:
            model.fit(recorder['train_data'], recorder['train_true'])
        else:
            print("There has some problem of recorder")
            return
        
        if 'test_data' in recorder:
            pres = model.predict(recorder['test_data'])
        else:
            print("There has some problem of recorder")
            return
        return model.predict(recorder['train_data']), pres,model
    
class L2_MLR(meta_model):
    def __init__(self, id=0, save=False, model_save_files="Stacking_model\\model_checkpoint") -> None:
        super().__init__(id, save, model_save_files)
        self.mark = "Multiple LinearRegression for the second layer of Stacking_model"
        self.name = "L2_MLR"
    
    def get_model(self,recorder: dict):
        print('Start fitting MLR in L2')
        model = LinearRegression()
        if 'train_data' in recorder and 'train_true' in recorder:
            model.fit(recorder['train_data'], recorder['train_true'])
        else:
            print("There has some problem of recorder")
            return
        
        if 'test_data' in recorder:
            pres = model.predict(recorder['test_data'])
        else:
            print("There has some problem of recorder")
            return
        return model.predict(recorder['train_data']), pres,model

class L2_SVR(meta_model):
    def __init__(self, id=0, save=False, model_save_files="Stacking_model\\model_checkpoint") -> None:
        super().__init__(id, save, model_save_files)
        self.mark = "Support vector regression for the second layer of Stacking_model"
        self.name = "L2_SVR"
    
    def get_model(self,recorder: dict):
        print('Start fitting SVR in L2')
        model = svm.SVR()
        if 'train_data' in recorder and 'train_true' in recorder:
            model.fit(recorder['train_data'], recorder['train_true'])
        else:
            print("There has some problem of recorder")
            return
        
        if 'test_data' in recorder:
            pres = model.predict(recorder['test_data'])
        else:
            print("There has some problem of recorder")
            return
        return model.predict(recorder['train_data']), pres,model
    
    
    
