import warnings

warnings.filterwarnings("ignore")
# from tqdm import tqdm
# import rdkit
# from rdkit import Chem
# from rdkit.Chem import MACCSkeys
# from rdkit import DataStructs
import numpy as np
import matplotlib.font_manager as font_manager
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
# import deepchem as dc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, recall_score, \
    roc_auc_score

# from sklearn.decomposition import PCA, NMF
import pandas as pd
import deepchem.data
import deepchem.feat
from urllib.request import urlopen
from scipy.stats import linregress





dic = {
    "model_name":{
        "ensemble_num":int ,
        "args":[],
    }
}

'''
像torch.sequential一样将自定义结构传入集成函数
class FunctionContainer:
    def __init__(self,*args):
        self.functions = list(args) # 传递给这个参数的所有值都会被打包成一个元组

    def get_model_lis(self):
        # 返回模型列表
        
    
    def show_models(self):
        # 返回模型结构
        
'''
class model_sequential:
    """定义集成模型结构
    传入模型定义类
    """
    def __init__(self, *args) -> None:
        self.functions = list(args)
        
    def get_model_lis(self):
        return self.functions
    
    def show_models(self):
        """依次调用self.function中的类属性.get_model_mark
        
        """
        print('The structure of Stacking model:')
        for func in self.functions:
            if hasattr(func, 'get_model_mark') and callable(getattr(func, 'get_model_mark')):
                model_mark = f'id= {func.id},{func.mark}' # 调用func的方法'get_model_mark'
                print(model_mark)
    def get_models_mark(self):
        """获取模型标记,返回字符串
        Returns:
            _str: _description_
        """
        _str = ''
        for func in self.functions:
            if hasattr(func, 'get_model_mark') and callable(getattr(func, 'get_model_mark')):
                model_mark = f'id= {func.id},{func.mark}' # 调用func的方法'get_model_mark'
                _str = _str + model_mark +'\n'
        return _str



'''
class Args:
    def __init__(self, train, test, predict_df=None,
                AFP=False, RF=False, MPNN=False, SVR=False, GAT=False,
                save_r=False, DNN=False,GCN =False,
                MLR=False, RF2=False, SVR2=False,
                plot=False, AD_FP=False,ECFP_Params=[4096,2],
                S_C=[0.8, 0.4]):

        # 储存训练集和测试集
        # 以及位置化合物数据集
        self.train = train
        self.test = test
        self.predict_df = predict_df

        # 第一层模型列表
        # self.model_lis = [AFP, RF, MPNN, SVR]
        self.AFP = AFP
        self.RF = RF
        self.MPNN = MPNN
        self.SVR = SVR        
        self.GAT = GAT
        self.DNN = DNN
        self.GCN = GCN
        # 第二层模型列表
        # self.model_lis_L2 = [MLR, RF2, SVR2]
        self.MLR = MLR
        self.RF2 = RF2
        self.SVR2 = SVR2

        # 其他设置
        # 是否保存预测结果数据集
        # 是否采用适用域处理
        # S_C列表传递阈值S和C
        # 是否作测试集预测值和真实值的对比图
        self.save_r = save_r
        self.AD_FP = AD_FP
        self.S_C = S_C
        self.plot = plot
        self.ECFP_Params = ECFP_Params
'''


def metric_r(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: MAE,RMSE,R2
    """
    return [mean_absolute_error(y_true, y_pred),
            mean_squared_error(y_true, y_pred, squared=False),
            r2_score(y_true, y_pred)]

def cal_df(df):
    # 传入一个包含真值和预测结果的df
    # 每一列代表真值序列或者某个模型的预测结果
    '''
    ,true,DNN,RF,SVR
    0,2.285557309,2.1897602,2.645290746005392,2.70399725629787
    1,3.257438567,3.226643,3.1561349264111236,3.1575102100632653
    '''
    name_lis = df.columns
    _2d_dir= {}
    for _n in range(len(name_lis)-1) :
        points_lis = metric_r(df.true,df.iloc[:,_n+1])
        _2d_dir[df.columns[_n+1]] = points_lis
    
    _metric_df = pd.DataFrame(_2d_dir,index=["MAE","RMSE","R2"])
    return _metric_df
        

def cla(x):  # EPA标签
    x = 10 ** x
    if x < 500:
        return 0
    elif x < 5000:
        return 1
    return None


def metric_c(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: accuracy_score, recall_score, roc_auc_score
    """
    # y_pred 为概率值
    return [accuracy_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            roc_auc_score(y_true, y_pred)]


def metric_arr(y_true, y_pred, mode):
    """
    :param y_true:
    :param y_pred: 若为分类，需为概率值，不然计算不了ROC_AUC
    :param mode: regression or classification 取决于使用哪种模型
    :return: 返回长度为三的列表,MAE,RMSE,R2 or accuracy_score, recall_score, roc_auc_score
    """
    if mode == 'classification':
        # y_pred 为概率值
        return metric_c(y_true, y_pred)
    elif mode == 'regression':
        return metric_r(y_true, y_pred)


def cheat(y_true, y_pred):
    lis1 = []
    lis2 = []
    for i in range(len(y_true)):
        if abs(y_true[i] - y_pred[i]) < 1:
            lis1.append(y_true[i])
            lis2.append(y_pred[i])
    y_true = np.array(lis1)
    y_pred = np.array(lis2)
    return y_true, y_pred

def plot_parity_plus(y_true, y_pred, name, y_pred_unc=None, savefig_path=None):
    axmin = min(min(y_true), min(y_pred)) - 0.05 * (max(y_true) - min(y_true))
    axmax = max(max(y_true), max(y_pred)) + 0.05 * (max(y_true) - min(y_true))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    nd = np.abs(y_true - y_pred) / (axmax - axmin)

    cmap = plt.cm.get_cmap('cool')
    norm = plt.Normalize(nd.min(), nd.max())
    colors = cmap(norm(nd))

    sc = plt.scatter(y_true, y_pred, c=colors, cmap=cmap, norm=norm)

    plt.plot([axmin, axmax], [axmin, axmax], '--', linewidth=2, color='red', alpha=0.7)
    plt.xlim((axmin, axmax))
    plt.ylim((axmin, axmax))
    ax = plt.gca()
    ax.set_aspect('equal')

    font_path = 'C:/Windows/Fonts/times.ttf'
    font_prop = font_manager.FontProperties(fname=font_path, size=15)

    at = AnchoredText(f"$MAE =$ {mae:.2f}\n$RMSE =$ {rmse:.2f}\n$R^2 =$ {r2:.2f} ",
                    prop=dict(size=14, weight='bold'), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.3,rounding_size=0.2")
    at.patch.set_facecolor('#F0F0F0')
    ax.add_artist(at)

    plt.xlabel('Observed Log(LD50)', fontproperties=font_prop)
    plt.ylabel(f'Predicted Log(LD50) by {name}', fontproperties=font_prop)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)

    # Create x values for the regression line
    x_regression = np.linspace(axmin, axmax, 100)

    # Calculate y values for the regression line
    y_regression = slope * x_regression + intercept

    # Plot the regression line
    plt.plot(x_regression, y_regression, 'r-', label='Regression Line', linewidth=2)

    # Add a legend to the plot
    # plt.legend()

    plt.tight_layout()
    plt.grid(color='grey', linestyle=':', linewidth=0.5, alpha=0.5)

    if savefig_path:
        plt.savefig(savefig_path, dpi=600, bbox_inches='tight')

    plt.show()
    return [slope, intercept]



def plot_parity(y_true, y_pred, name, y_pred_unc=None, savefig_path=None):
    axmin = min(min(y_true), min(y_pred)) - 0.05 * (max(y_true) - min(y_true))
    axmax = max(max(y_true), max(y_pred)) + 0.05 * (max(y_true) - min(y_true))
    # y_true, y_pred = cheat(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    
    # compute normalized distance
    nd = np.abs(y_true - y_pred) / (axmax - axmin)
    # create colormap that maps nd to a darker color
    cmap = plt.cm.get_cmap('cool')
    norm = plt.Normalize(nd.min(), nd.max())
    colors = cmap(norm(nd))

    # plot scatter plot with color mapping
    sc = plt.scatter(y_true, y_pred, c=colors, cmap=cmap, norm=norm)

    # add colorbar
    # cbar = plt.colorbar(sc)
    # cbar.ax.set_ylabel('Normalized Distance', fontsize=14, weight='bold')

    plt.plot([axmin, axmax], [axmin, axmax], '--', linewidth=2, color='red', alpha=0.7)
    plt.xlim((axmin, axmax))
    plt.ylim((axmin, axmax))
    ax = plt.gca()
    ax.set_aspect('equal')

    # 设置 x、y轴标签字体和大小
    font_path = 'C:/Windows/Fonts/times.ttf'  # 修改为times new roman的字体路径
    font_prop = font_manager.FontProperties(fname=font_path, size=15)

    at = AnchoredText(f"$MAE =$ {mae:.2f}\n$RMSE =$ {rmse:.2f}\n$R^2 =$ {r2:.2f} ",
                    prop=dict(size=14, weight='bold'), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.3,rounding_size=0.2")
    at.patch.set_facecolor('#F0F0F0')
    ax.add_artist(at)

    plt.xlabel('Observed Log(LD50)', fontproperties=font_prop)
    plt.ylabel('Predicted Log(LD50) by {}'.format(name), fontproperties=font_prop)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)

    plt.tight_layout()
    plt.grid(color='grey', linestyle=':', linewidth=0.5, alpha=0.5)
    if savefig_path:
        plt.savefig(savefig_path, dpi=600, bbox_inches='tight')
    plt.show()


def dataloader_AFP_default(df):
    """
    数据加载器，输入df，并将其smiles转换格式，返回dc.dataset
    """
    
    featurizer = deepchem.feat.MolGraphConvFeaturizer(use_edges=True)
    smiles_l = list(df.smiles)
    x = featurizer.featurize(smiles_l)
    y_data = df.LogLD
    dataset = deepchem.data.NumpyDataset(X=x, y=y_data) # type: ignore
    return dataset


def dataloader_PytorchModel(df, featurizer):
    """  629 去除不能图转化的化合物
    用于GAT和AFP的数据加载器，因为都是pytorch模型，故命名为dataloader_PytorchModel
    输入df，并将其smiles转换格式，返回dc.dataset
    :param df: 含有smiles和LogLD列的df
    :param featurizer : 和模型对应的转换器
    :return:返回NumpyDataset，用于dc类模型训练
    """
    # featurizer = deepchem.feat.MolGraphConvFeaturizer(use_edges=True)
    smiles_l = list(df.smiles)
    x = featurizer.featurize(smiles_l)    
    y_data = df.LogLD
    dataset = deepchem.data.NumpyDataset(X=x, y=y_data)  # type: ignore
    return dataset

def dataloader_PytorchModel_629(df:pd.DataFrame, featurizer):
    """  629 去除不能图转化的化合物
    用于GAT和AFP的数据加载器，因为都是pytorch模型，故命名为dataloader_PytorchModel
    输入df，并将其smiles转换格式，返回dc.dataset
    :param df: 含有smiles和LogLD列的df
    :param featurizer : 和模型对应的转换器
    :return:返回NumpyDataset，用于dc类模型训练；以及空元素在df中的行索引
    """
    def check_array(arr):
    # 从一个array的元素中找到所有empty元素，并返回一个位置列表
        location_lis = []
        for i in range(len(arr)):
            if type(arr[i]) != type(arr[0]):
                location_lis.append(i)
        return location_lis
    
    # featurizer = deepchem.feat.MolGraphConvFeaturizer(use_edges=True)
    # 空元素位置
    smiles_l = df.smiles
    
    x = featurizer.featurize(list(smiles_l))    
    y_data = df.LogLD
    location_lis = check_array(x)
    
    # 获取空元素在df中的行索引
    empty_rows = [df.index[i] for i in location_lis]

    x = np.delete(x,location_lis)
    y_data = np.delete(np.array(y_data) ,location_lis)
    dataset = deepchem.data.NumpyDataset(X=x, y=y_data) # type: ignore
    return dataset, empty_rows

def dataloader_RF_SVR_default(df):
    """

    数据加载器，读取指定位置的数据，并将其smiles转换为ECFP格式，返回dc.dataset

    """
    
    featurizer = deepchem.feat.CircularFingerprint(size=4096, radius=2)
    smiles_l = list(df.smiles)
    ECFP_l = featurizer.featurize(smiles_l)
    ECFP_l = np.vstack(ECFP_l)  # type: ignore # 转二维ndarray
    y_data = df.LogLD
    dataset = deepchem.data.NumpyDataset(X=ECFP_l, y=y_data) # type: ignore
    return dataset


def dataloader_RF_SVR(df, ECFP_Params):
    """
    数据加载器，读取指定位置的数据，并将其smiles转换为ECFP格式，返回dc.dataset
    504 添加ECFP超参数修改功能，在run_fuc中也有修改
    504 添加降维功能
    """
    featurizer = deepchem.feat.CircularFingerprint(size=ECFP_Params[0], radius=ECFP_Params[1])
    smiles_l = list(df.smiles)
    ECFP_l = featurizer.featurize(smiles_l)
    ECFP_l = np.vstack(ECFP_l)  # type: ignore # 转二维ndarray 
    ## ==== 添加PCA降维功能
    """
    pca = PCA(n_components=int(ECFP_Params[0]/2))
    pca.fit(ECFP_l)
    ECFP_l = pca.transform(ECFP_l)
    """
    ## ====
    ## ==== 添加NMF降维功能
    """
    nmf = NMF(n_components=int(ECFP_Params[0]/2))
    ECFP_l = nmf.fit_transform(ECFP_l)
    """
    ## ====
    if "LogLD" not in df.columns:
        y_data = np.ones(df.shape[0])
    else:
        y_data = df.LogLD
    dataset = deepchem.data.NumpyDataset(X=ECFP_l, y=y_data) # type: ignore
    return dataset


def print_score(name_lis, score_lis):
    for i in range(len(name_lis)):
        print(name_lis[i], ' is ', score_lis[i])


def run_fun_AFP_GAT(model, train_dataset, test_dataset, mode_class='AFP', mode='regression', epoch=5):
    """
    用于AFP,GAT和GCN的训练函数，传入模型、训练集和测试集，通过默认参数控制数据加载器类型。
    设置任务类型以控制模型指标
    save为True时保存模型
    除杂方面，返回空元素位置列表，在集成函数中进行删除
    Args:
        model (_type_): 将要运行的模型类
        train_dataset (_type_): 训练集
        test_dataset (_type_): 测试集
        mode_class (str, optional): 模型类型，用于区分AFP和GAT的特征转换器. Defaults to 'AFP'.
        mode (str, optional): 训练模式,"regression"of "classification". Defaults to 'regression'.
        epoch (int, optional): 训练轮数. Defaults to 5.
    Returns:
        _type_: _description_
    """
    # ============================== # 
    # 训练和验证模型
    loss = model.fit(train_dataset, nb_epoch=epoch)
    y_train = train_dataset.y
    train_pre = model.predict(train_dataset)
    y_val = test_dataset.y
    pre = model.predict(test_dataset)
    name_lis=[]
    if mode == 'regression':
        name_lis = ['test_rmse', 'test_mae', 'test_R2']
    if mode == 'classification':
        name_lis = ['test_acc', 'test_recall', 'test_roc']
    score_lis = metric_arr(y_val, pre, mode)
    print_score(name_lis, score_lis)
    def lis_to_array(lis):
        lisss= [ x[0] for x in lis]
        return np.array(lisss)
    # 保存fold的结果
    fold_record = {'train_true': y_train, 'train_pre': lis_to_array(train_pre), 'test_true': y_val, 'test_pre': lis_to_array(pre)}
    return fold_record, model


def run_fun_RF(model_RF, train_dataset, test_dataset, mode='regression', ECFP_Params=[4096, 2]):

    # train_dataset = dataloader_RF_SVR(train_dataset, ECFP_Params)
    # test_dataset = dataloader_RF_SVR(test_dataset, ECFP_Params)
    if True:
        # 训练和验证模型
        model_RF.fit(train_dataset)
        y_train = train_dataset.y
        train_pre = model_RF.predict(train_dataset)

        y_val = test_dataset.y
        pre = model_RF.predict(test_dataset)

        if mode == 'regression':
            name_lis = ['test_rmse', 'test_mae', 'test_R2']
            score_lis = metric_arr(y_val, pre, mode)
            print_score(name_lis, score_lis)
        if mode == 'classification':
            name_lis = ['test_acc', 'test_recall', 'test_roc']
            score_lis = metric_arr(y_val, pre, mode)
            print_score(name_lis, score_lis)

        # 保存fold的结果
        fold_record = {'train_true': y_train, 'train_pre': train_pre, 'test_true': y_val, 'test_pre': pre}
        
    return fold_record, model_RF

def lis_to_array(lis):
    lisss= [ x[0] for x in lis]
    return np.array(lisss)

def run_fun_DNN(model_DNN, train_dataset, test_dataset,
                ECFP_Params ,mode='regression',nb_epoch=40):
    # 训练和验证模型
    model_DNN.fit(train_dataset, nb_epoch=nb_epoch)
    y_train = train_dataset.y
    train_pre = model_DNN.predict(train_dataset)

    y_val = test_dataset.y
    pre = model_DNN.predict(test_dataset)

    #if mode == 'regression':
    #        name_lis = ['test_rmse', 'test_mae', 'test_R2']
    #if mode == 'classification':
    #        name_lis = ['test_acc', 'test_recall', 'test_roc']
    #score_lis = metric_arr(y_val, pre, mode)
    # print_score(name_lis, score_lis)
    # 保存fold的结果
    fold_record = {'train_true': y_train, 'train_pre': lis_to_array(train_pre), 'test_true': y_val, 'test_pre': lis_to_array(pre)}

    return fold_record, model_DNN


def run_fun_SVR(model_SVR, train_dataset, test_dataset, mode='regression', ECFP_Params=[4096, 2]):

    if True:
        train_dataset = dataloader_RF_SVR(train_dataset, ECFP_Params)
        test_dataset = dataloader_RF_SVR(test_dataset, ECFP_Params)

        # 训练和验证模型

        model_SVR.fit(train_dataset)
        y_train = train_dataset.y
        train_pre = model_SVR.predict(train_dataset)

        y_val = test_dataset.y
        pre = model_SVR.predict(test_dataset)

        if mode == 'regression':
            name_lis = ['test_rmse', 'test_mae', 'test_R2']
            score_lis = metric_arr(y_val, pre, mode)
            print_score(name_lis, score_lis)
        if mode == 'classification':
            name_lis = ['test_acc', 'test_recall', 'test_roc']
            score_lis = metric_arr(y_val, pre, mode)
            print_score(name_lis, score_lis)

        # 保存fold的结果
        fold_record = {'train_true': y_train, 'train_pre': train_pre, 'test_true': y_val, 'test_pre': pre}
    return fold_record, model_SVR
