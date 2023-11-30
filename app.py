from shiny import App, render, ui, reactive
import shinyswatch

# -----------------------------------------------------
# from ipydatagrid import DataGrid
# from shinywidgets import output_widget, render_widget, register_widget

from shiny.types import NavSetArg
import os
import zipfile
from typing import List
from pathlib import Path    # for 'www'
import json
import numpy as np
import pandas as pd
import datetime
from datetime import date
# -----------------------------------------------------
# self-defined module
from DB.DB import db
import models.utils as utils
from htmltools import css





# import models.meta_models as meta_models



#####
'''
使用延迟导入的方式优化程序启动
'''
# import models.LAY2 as LAY
import importlib

def lazy_import(module_name):
    return importlib.import_module(module_name)

#####



def  home_tags_list():
    return [
        ui.tags.div(
            ui.tags.img(src="img/stack.png", 
                        class_ = "img-fluid mx-auto d-block", style ="max-width: 95%;"),
            ui.row(
                ui.column(12, 
                        ui.tags.p("""
                                We evaluated five machine learning algorithms based on molecular descriptors for addressing the problem of compound toxicity prediction: Random Forest (RF), Support Vector Regression (SVR), Deep Neural Network (DNN), Gradient Boosting Tree (GBM), and Adaboost Boosting Tree (ADA), as well as four graph neural network models based on graphs: Directed Message Passing Neural Network (D-MPNN), Attentive FP model, Graph Convolutional Neural Network (GCN), and Graph Attention Network (GAT). A three-layer Stacking ensemble model was constructed using the Super Learner method.
                                """
                                ),
                        ),
            ),     
            class_ = "container"   
        )
        # ui.tags.p("我们评估了基于分子描述符的五种机器学习算法 [随机森林 (RF)，支持 向量回归 (SVR)，深度学习模型 (DNN)，梯度提升树 (GBM)，Adaboost 提升树 (ADA) 以及基于图的三种图神经网络 [信息传递神经网络 (D-MPNN), Attentive FP 模型, 图卷积神经网络 GCN] 来解决化合物毒性预测的问题。使用超级学习器（Super learner）方法搭建了三层的 Stacking 集成模型，作为毒性预测的QSAR模型。考虑位于活性悬崖 (ACs) 中的化合物，基于分子描述符和目标特征，给出了表征模型适用域 AD 的方法。"),
    ]

L1_choices = {"a":"RF","b":"SVR","c":"DNN","d":'MPNN',"e":"AFP", }
L2_choices = {"a":"L2_RF","b":"L2_SVR","c":"L2_MLR"}
ECFP_l_choices ={"a":"1024","b":"2048","c":"4096"}
ECFP_r_choices ={"a":"2","b":"4","c":"6","d":"8"}
def  training_tags_list():    
    return [
        ui.layout_sidebar(
            # 页面侧边栏
            ui.panel_sidebar(
                ui.markdown("""
                            ## Stacking Model Usage Instructions
                            
                            This platform supports users to train their own models using their datasets, thus obtaining 
                            machine learning models suitable for a specific type of toxicity data. The platform supports 
                            building a three-layer ensemble model using the Stacking method, and currently, the first layer 
                            models can be chosen from five machine learning methods.
                            
                            Here are the steps to follow:
                            
                            1. Determine the number of layers you want to use and decide on the types of models to be used in each layer. 
                            It is generally recommended to use three layers. 
                            (Based on experimental results, the second and third layers do not introduce significant computational costs but can greatly improve prediction performance.)
                            2. Submit compound toxicity data. You can also choose to submit test data to assess model performance. 
                            The data should be submitted in a '''csv''' format following the provided example style. 
                            3. Start the training process. After training is completed, you can view the evaluation results, download prediction outcomes, and access the trained model.
                            4. Download pretrained model for further prediction.
                            """),
                ),
                
                #本平台支持用户采用自己的数据集训练个人的模型，从而得到适用于一类毒性数据的机器学习模型。
                #本平台支持适用Stacking方法搭建三层的集成模型，目前第一层模型支持使用五种机器学习方法。
                
                #使用步骤如下：
                #1. 需要选用的模型层数并且确定每层选用的模型种类，一般推荐使用三层。（因为从实验结果来看，第二、三层并不会带来过高的计算成本，但是可以显著提高预测效果）。
                #2. 提交化合物毒性数据，也可以选择提交测试数据，来检测模型性能。。数据需要按照示例样式，以```csv```的格式提交，其中需包含smiles，cas和毒性数据三列。
                #3. 开始训练，训练结束后可以查看评估结果，下载预测结果以及模型。
                
            # 页面主内容
            ui.panel_main(
                ui.navset_tab_card(
                    # 1.1 模型结构
                    ui.nav("Training Step 0: Demo",
                        ui.p("你可以下载此处demo:"),
                        ui.row(
                            ui.column(5,ui.download_button("download_train_demo","训练集demo下载")),
                            ui.column(5,ui.download_button("download_test_demo","测试集demo下载")),
                        ),
                        ui.markdown("""
                                    推荐使用如下模型结构：
                                    
                                    id= 0,Random Forest Regressor useing ECFP
                                    
                                    id= 1,deepchem.models.AttentiveFPModel:Model for Graph Property Prediction
                                    
                                    id= 2,Deep neural network useing ECFP
                                    
                                    id= 3,Support vector regression useing ECFP
                                    
                                    ***
                                    The structure of Stacking model's second layer :
                                    
                                    id= 0,Support vector regression for the second layer of Stacking_model
                                    
                                    id= 1,Random Forest Regressor for the second layer of Stacking_model
                                    
                                    id= 2,Support vector regression for the second layer of Stacking_model
                                    
                                    ***
                                    
                                    第三层使用nnls
                                    """),

                        ),
                    
                    ui.nav("Training Step 1: Select Model Structure",
                        ui.row(
                            ui.column(6,
                                # 1.1.1 模型结构选择
                                ui.div(
                                    ui.div(
                                        {"class":"card",
                                        # 好像没用"style":"position:relative,left:110px;justify-content: center"
                                        },
                                        ui.p("Model selection"),
                                        # 模型数量选择
                                        ui.div(
                                            ui.input_select("num_meta_model", "The number of models in the first layer",["1","2","3","4","5","6"]
                                            ),
                                            style = css(position="relative",left="10px"), # type: ignore 
                                        ),
                                        
                                        # 第一层模型选择，需要根据num_meta_model,弹出相应数量的选择框
                                        ui.div(                                    
                                            ui.input_select("L1_choices_0", "Model_0 selection in the first layer", L1_choices,),
                                            
                                            ui.panel_conditional("['2','3','4','5','6'].includes(input.num_meta_model)",
                                                ui.input_select("L1_choices_1", "Model_1 selection in the first layer", L1_choices,),
                                            ),
                                            ui.panel_conditional("['3','4','5','6'].includes(input.num_meta_model)",
                                                ui.input_select("L1_choices_2", "Model_2 selection in the first layer", L1_choices,),
                                            ),
                                            ui.panel_conditional("input.num_meta_model === '4' || input.num_meta_model === '5' || input.num_meta_model === '6'",
                                                                ui.input_select("L1_choices_3", "Model_3 selection in the first layer", L1_choices,),
                                                                ) ,
                                            ui.panel_conditional("input.num_meta_model === '5' || input.num_meta_model === '6'",
                                                                ui.input_select("L1_choices_4", "Model_4 selection in the first layer", L1_choices,),
                                                                ) ,
                                            ui.panel_conditional("input.num_meta_model === '6' ",
                                                                ui.input_select("L1_choices_5", "Model_5 selection in the first layer", L1_choices,),
                                                                ) ,
                                            style = css(position="relative",left="10px"),
                                        ),
                                        
                                    ),
                                    ui.p(), # 空行
                                    
                                    ui.div(
                                        {"class":"card"},
                                        ui.div(
                                            ui.div(ui.input_switch("L2", "Whether to use L2"),style = css(position="relative",top="10px"),value=False),
                                            ui.panel_conditional("input.L2" , 
                                                ui.input_checkbox_group("L2_choices", "Model selection in the secend layer", L2_choices,selected=['a','b','c']),
                                                ui.input_switch("L3_choices", "Whether to use L3",False),
                                            ),
                                        style = css(position="relative",left="10px"),
                                        )
                                    )

                                )
                            ),
                            ui.column(6,
                                # 1.1.2 模型参数选择
                                ui.div(
                                    {"class":"card"},
                                    ui.markdown("""
                                                ### Molecular Descriptor Selection
                                                
                                                For machine learning algorithms beyond graph neural networks, appropriate parameters need to be selected. 
                                                Currently, this project supports the use of Extended-Connectivity Fingerprint, which requires pre-setting two parameters:
                                                1. Fingerprint Length : Specify the length of the generated extension fingerprints, which determines the number of bits included. 
                                                2. Fingerprint Radius: Specify the neighborhood range of atoms in the extension fingerprints.
                                                
                                                """
                                    ),
                                    # 对于图神经网络之外的机器学习算法需要选择合适的参数，目前本项目支持使用分子拓展链接指纹，这种算法需要预设两种参数：
                                    # 1. 拓展指纹长度（length）：指定生成的拓展指纹的长度，即包含多少个位（bits）。较长的拓展指纹可以提供更多的结构信息，但也会增加计算复杂性。
                                    # 2. 半径（radius）：指定原子在拓展指纹中的邻居范围。
                                    ui.div(
                                        ui.input_select("ECFP_l"," Fingerprint Length",ECFP_l_choices),
                                        ui.input_select("ECFP_r","Radius",ECFP_r_choices),
                                        style = css(position="relative",left="10px"),
                                    )
                                )
                            )    
                        ),
                    ),

                    # 1.2：数据提交
                    ui.nav("Training Step 2: Submit compound toxicity data",
                        # ui.tags.p("训练步骤二：上传数据"),
                        ui.row(
                            ui.column(6,
                                ui.div(    
                                    # {"position":"relative","left":"10px","top":"5px"},
                                    # 1.2.1 样例数据展示
                                    
                                    ui.row(
                                        ui.column(4,
                                            ui.input_switch("sample_data","show exemple data",True),
                                        ),
                                        ui.column(2),
                                        ui.column(4,
                                            ui.download_button("download_exemple_df","download exemple data"),
                                        )
                                    ) ,
                                    ui.panel_conditional("input.sample_data",ui.output_table("exemple_data"),), 
                                    
                                ),
                                
                            ),
                            ui.column(1),
                            ui.column(5,
                                {"class":"card"},
                                ui.div(
                                    
                                    # 1.2.2 用户数据上传
                                    # 上传训练数据
                                    ui.input_file("training_set", "Please upload the training dataset:", multiple=False),
                                    # 上传测试集
                                    ui.input_checkbox("if_test_data","Whether to use the test dataset"),
                                    ui.panel_conditional("input.if_test_data",
                                                        ui.panel_conditional("input.if_test_data",
                                                        ui.input_file("test_set", "Please upload your test dataset:", multiple=False),
                                                            ),
                                                        )
                                ),
                            )

                        )
                                                
                    ),
                    # 1.3 模型训练页面
                    ui.nav("Training Step 3: Start Training",
                        ui.div(
                            {"class":"card"},
                            
                            ui.tags.p("Click the button to start the training process. It may take some time.",
                                    style = css(position="relative",left="10px") ),
                            # 1.3.1 开始训练按钮
                            ui.input_action_button("do_training", "Training...",
                                                width="400px",
                                                style = css(position="relative",left="10px"),
                            ),
                            ui.p(),
                            # 1.3.2 模型结果展示，测试和训练结果评价
                            ui.input_checkbox("train_plot","show train metric table",False,
                                            ),
                            ui.panel_conditional("input.train_plot" , 
                                                ui.output_table("show_train_metric_table")
                                ) ,
                            ui.input_checkbox("test_plot","show test metric table",False),
                            ui.panel_conditional("input.test_plot" ,
                                                ui.output_table("show_test_metric_table"),
                                                ),
                            # 1.3.3 模型结果下载
                            ui.panel_conditional("input.do_training",
                                ui.output_text_verbatim("training_result_txt"),
                                ui.row(
                                    ui.column(6,ui.download_button("dl_result_train","Model Results Download: Training Set",class_="btn-info")),
                                    ui.column(6,
                                        ui.panel_conditional("input.if_test_data",
                                            ui.download_button("dl_result_test","Model Results Download: Test Set",class_="btn-info")
                                        ),
                                    ),
                                )                                       
                            ),
                        )

                    ),
                    
                    # 1.4 打包模型以及数据
                    ui.nav("Training Step 4: Download",
                        ui.tags.p("Before the next prediction step, you need to download the pre-trained model."), #要进行下一步预测操作前，需要自行下载训练好的模型。
                        ui.download_button("model_zip_download","Download the packaged model.",width="400px"), # 下载打包好的模型
                    ),
                    id = "training_tabs",
                    ),
            ),
        ),
    ]

def  predict_tags_list():
    return [
    ui.layout_sidebar(
        ui.panel_sidebar(
            # {"class":"card"},
            # ui.input_radio_buttons("type", "Prediction Model:", 
            #                        ["Built-in", "Pre-training"], 
            #                        selected = "Pre-training",
            #                        inline=True),  
            
            #ui.input_file("predict_set", "Please upload the dataset to predict:", multiple=False),

            #ui.input_action_button("do_predict", "Predict...",  class_="btn-primary"),
            ui.markdown('''
                        ## Stacking Model Prediction Usage Instructions
                        For the toxic data that needs prediction, you can choose to use our pre-trained models (specifically designed for oral rat LD_{50}), or you can use the models that you trained on your own data in the previous step.
                        '''), # 对于需要进行预测的化合物数据，可以选择使用我们预训练好的模型（针对小鼠口服毒性的模型），或者使用您在上一步骤中在您自己的数据上训练好的模型。
            ui.input_select("model_check","Mode selection",choices={"a":"upload your model","b":"our pre-trained models for oral rat LD50"},selected="b"),
            ui.panel_conditional("input.model_check == \"a\" ", 
                                ui.input_file("users_model","upload your model",accept=".zip"),
                                ui.input_action_button("zipping","upload"),
                                ui.output_text_verbatim("finish_zip"),
                                ),
            
            ui.panel_conditional("input.model_check == \"b\" ", 
                                ui.markdown('''
                                            The structure of our pre-trained models:
                                            * The first layer：[RF,SVR,DNN]
                                            * The second layer[RF2,SVR2,MLR]
                                            * The third layer[nnls]
                                            * ECFP_length = 4096 , ECFP_r = 2
                                            ''')
                                ),
                                # ui.output_text_verbatim("finish_zip"),
        ),
        ui.panel_main(
            ui.navset_tab_card(
                ui.nav("Step 2: Submit compound toxicity data for prediction",
                    ui.div(
                        ui.row(
                            ui.column(6,
                                {"class":"card"},
                                # 展示示例数据
                                ui.p(),
                                ui.row(
                                    ui.column(4,
                                        ui.input_switch("sample_testdata","show exemple data",True),
                                    ),
                                    ui.column(1),
                                    ui.column(4,
                                        ui.download_button("download_exemple_predf","download exemple data"),
                                    )
                                ),
                                
                                ui.panel_conditional("input.sample_testdata",ui.output_table("sample_test"),), 
                            ),
                            ui.column(6,
                                {"class":"card"},
                                ui.p(),
                                # 上传训练数据
                                ui.input_file("predict_set", "Please upload the predict dataset:", multiple=False),
                                )
                        )
                    )
                ),
                ui.nav("Step 3 : The prediction process", 
                    ui.div(
                        {"class":"card"},
                            
                        ui.tags.p("Click the button to start the prediction process. It may take some time.",
                                style = css(position="relative",left="10px") ),

                        ui.input_action_button("do_predicting", "predicting...",
                                                width="400px",
                                                style = css(position="relative",left="10px"),
                        ),
                        ui.p(),
                        ui.output_text_verbatim("pre_finish_text"),
                    ),
                    ui.div(
                        {"class":"card"},
                        ui.p(),
                        ui.download_button("predict_result_download","Download the results.",width="400px"),
                        ui.p(),
                    )
                ),
                id = "predict_tabs",
            ),
        width = 9,

        ),
    ),

    ]


def  about_tags_list():
    return [
        
    ]


def nav_controls(prefix: str) -> List[NavSetArg]:
    return [
        ui.nav("Home", *home_tags_list()),
        ui.nav("Training", 
            *training_tags_list(),
        ),
        ui.nav("Prediction", *predict_tags_list()),
        ui.nav("About", *about_tags_list()),
        ui.nav_spacer(),  
        ui.nav_control(
            ui.a(
                'github',
                href="https://github.com/donkeyEEE/Stacking_model",
                target="_blank",
            ),

        ),

    ]

app_ui = ui.page_navbar(
    shinyswatch.theme.materia(),# One-line styling with Shinyswatch
    *nav_controls("page_navbar"),
    title="Stacking Model",
    bg="#0062cc",
    inverse=True,
    id="navbar_id",
)

exemple_df = pd.read_csv(Path(__file__).parent/"www/files/sample.csv")


def server(input, output, session):
    
    _trained = reactive.Value(False)
    
    import random
    # 生成随机数
    def random_char(length):
        string=[]
        for i in range(length):
            x = random.randint(1,2)
            if x == 1:
                y = str(random.randint(0,9))
            else:
                y = chr(random.randint(97, 122))
            string.append(y)
        string = ''.join(string)
        return string
    _number = random_char(20)
    print(_number)
    
    # 多用户管理系统
    # 在本地新建_number栈，容量为5，新用户使用时入栈，栈满时，去除栈顶元素
    def stack_management(number):
        max_files = 3
        # 栈以json的形式保存,若存在栈则读取，若不存在栈则新建立
        if os.path.exists('keys.json'):
            with open('keys.json', 'r') as file:# 从文件加载列表
                number_lis = json.load(file)
            number_lis.append(number)
            if len(number_lis) > max_files:
                delet_num = number_lis.pop(0)
                cleaner = clean_the_file()
                cleaner(f'moldel_checkpoint_{delet_num}') # 删除多余文件
        else:
            number_lis = [number]
            with open('keys.json', 'w') as file:
                json.dump(number_lis, file)
    
    stack_management(_number)
    
    @reactive.Calc
    def get_training_db():
        file_infos = input.training_set()

        return db(file_infos[0]["datapath"])
    
    @output
    @render.table # type: ignore
    def training_table():
        df_lis = get_training_db().data
        if df_lis == None:
            return pd.DataFrame([])
        return df_lis
    
        # file_infos is a list of dicts; each dict represents one file. Example:
        # [
        #   {
        #     'name': 'data.csv',
        #     'size': 2601,
        #     'type': 'text/csv',
        #     'datapath': '/tmp/fileupload-1wnx_7c2/tmpga4x9mps/0.csv'
        #   }
        # ]
        
        
    @reactive.Calc
    def get_predict_db():

        file_infos = input.test_set()
        if not file_infos:
            return None
        
        print(file_infos)
        
        return db(file_infos[0]["datapath"])
    
    @reactive.Calc
    def get_predict_df():
        df = get_predict_db()

        if df is None:
            # return pd.DataFrame()
            return None

        return df.data  # .iloc[input.range_observations()[0] - 1:input.range_observations()[1], :]    
    
    @output
    @render.table
    def predict_table():
        df_lis = get_predict_df()
        return df

    # ========================
    @reactive.Effect
    @reactive.event(input.do_training)
    def _():
        ui.update_navs("training_tabs", selected="Training Result")

    @reactive.Effect
    @reactive.event(input.do_predict)
    def _():
        ui.update_navs("predict_tabs", selected="Predict Result")

    # ===================
    
    @reactive.Calc
    def clean_the_file():
        """ 删除model_checkpoints中原有数据
        """
        def clear_folder(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    clear_folder(file_path)
                    os.rmdir(file_path)
        return clear_folder
    
    @reactive.Calc
    def clean_metric_file():
        if os.path.isfile(f'model_checkpoint_{_number}/train_metric.csv'):
            os.remove(f'model_checkpoint_{_number}/train_metric.csv')
        
        if os.path.isfile(f'model_checkpoint_{_number}/test_metric.csv'):
            os.remove(f'model_checkpoint_{_number}/test_metric.csv')
        
    #获取根据输入获得模型结构
    @reactive.Calc
    @reactive.event(input.do_training) # Take a dependency on the button
    def get_model_sequential() -> utils.model_sequential:
        """解析输入获得模型结构

        Returns:
            utils.model_sequential: _description_
        """
        num_models = int(input.num_meta_model())
        lis = []
        ECFP_Params=[int(ECFP_l_choices[input.ECFP_l()]) ,int(ECFP_r_choices[input.ECFP_r()])]
        meta_models = lazy_import('models.meta_models')
        for i in range(num_models):
            # 获取选项
            # 可能出现调用input属性问题
            _m = getattr(input,f'L1_choices_{i}')()

            #if i == 1:
            #    _m = input.L1_choices_1
            name_model = L1_choices[_m]
            print(name_model)
            model = getattr(meta_models,f'meta_{name_model}')()
            if hasattr(model,'ECFP_Params'):
                setattr(model,'ECFP_Params',ECFP_Params)
                setattr(model,'id',i)
            lis.append(model)
            
        return utils.model_sequential(*lis)
    
    @reactive.Calc
    @reactive.event(input.do_training) # Take a dependency on the button
    def get_L2model_sequential() -> utils.model_sequential:
        """解析输入获得模型结构
        Returns:
            utils.model_sequential: _description_
        """
        lis = []
        meta_models = lazy_import('models.meta_models')
        for _choices in input.L2_choices():
            name_model = L2_choices[_choices]
            model = getattr(meta_models,f'{name_model}')()
            lis.append(model)
        return utils.model_sequential(*lis)
    
    # ======
    
    # 开始训练模型
    # 获取模型，并且可以通过ensemble_models.Lx_train_df
    # 查询df
    @reactive.Calc
    @reactive.event(input.do_training) # Take a dependency on the button
    def get_model():
        check_path= f'model_checkpoint_{_number}'
        # return None
        # print(int(input.num_meta_model()))
        
        num_p =3
        if input.L2:
            num_p = 4
        elif input.L3_choices():
            num_p = 5
        with ui.Progress(min=1, max=num_p ) as p:
            p.set(message="Calculation in progress", detail="This may take a while...")
            # 1.1 获取训练集和测试集数据
            dataloader = get_training_db()
            train_df = dataloader.data # type: ignore 
            p.set(1, message="Get training & test dataset")
            dataloader = get_predict_db()
            test_df = dataloader.data   # type: ignore 
            
            # print(train_df)
            # 进度条
            
            p.set(2, message="Level 1 training...")
            # 1.2 获取model_sequential
            # MPNN尚不支持py中运行，GAT和DNN不支持保存模型以及对未知数据进行预测
            model_struture = get_model_sequential()
            model_struture.show_models()
            print(model_struture)
            # print(f"ssssss{model_struture}")
            # 1.3实例化集成模型
            LAY = lazy_import('models.LAY2')
            eb = LAY.ensemble_models_new(model_save_files=check_path)  
            # if len(input.L1_choices()) ==0:
            #    return

            # 1.4 训练第一层模型，save选择是否保存模型
            eb.train_meta_models(train_df , test_df ,model_squential=model_struture,save=True)
            
            # 1.5 训练第二层模型，save选择是否保存模型
            # eb.L2_training()
            if input.L2():
                p.set(3, message="Level 2 training...")
                model_struture = get_L2model_sequential()
                eb.L2_training(model_struture,save=True)
                
            # 1.6 训练第三层模型，save选择是否保存模型
            # 第三层模型为非负最小二乘，只有保存权重和偏置
            
            if input.L3_choices() and input.L2():
                eb.L3_training(save=True)
                p.set(4, message="Level 3 training...")
            

            p.set(num_p, message="Done!")
        train_metric_table()
        test_metric_table()
        return eb
        
    
    # =====
    @output
    @render.table
    def exemple_data():
        return exemple_df
    
    @session.download()
    def download_exemple_df():
        return str(Path(__file__).parent/"www/files/sample.csv")
    
    # =====
    
    # 训练主函数
    @output
    @render.text
    @reactive.event(input.do_training) # Take a dependency on the button
    def training_result_txt():
        
        # 删除tmp中文件,如果存在临时文件夹就先清除
        file_path = f'model_checkpoint_{_number}'
        if os.path.exists(file_path):
            cleaner = clean_the_file()
            cleaner(file_path)
        else:
            pass 
        
        # 删除原有的评估数据
        clean_metric_file()
        
        model = get_model()

        if model == None:
            return None

        now_time = datetime.datetime.now()

        # set the flag that the model is ready to predict
        _trained.set(True)

        # 生成新的评估数据
        
        
        return f"Done！{now_time}"
    
    # =====
    
    # =====
    # 下载demo
    @session.download(
        # filename=lambda: f"data-{date.today().isoformat()}-{np.random.randint(100,999)}.csv"
    )
    def download_train_demo():
        return str(Path(__file__).parent/"www/files/train_demo.csv")
    
    @session.download(
        # filename=lambda: f"data-{date.today().isoformat()}-{np.random.randint(100,999)}.csv"
    )
    def download_test_demo():
        return str(Path(__file__).parent/"www/files/test_demo.csv")
    
    # 获取结果数据
    # 训练详细数据下载
    @session.download(
        # filename=lambda: f"data-{date.today().isoformat()}-{np.random.randint(100,999)}.csv"
    )
    def dl_result_train():
        # model = get_model()
        # if model == None:
        #    return None
        # filename = f"train-{date.today().isoformat()}-{np.random.randint(1000,9999)}.csv"
        # model.L3_train_df.to_csv(filename, index=False, header=True)
        
        def _():
            # 将三层的数据放到三张sheet上
            # 创建一个Excel写入器，用于将数据保存为一个带有三个sheet的CSV文件
            path = str(Path(__file__).parent/"model_checkpoint")
            writer = pd.ExcelWriter(f'{path}/all_train_data.xlsx')
            
            # 将每个CSV数据写入不同的sheet
            csv1 = pd.read_csv(f'{path}/L1_train_data.csv')
            csv1.to_excel(writer, sheet_name='Sheet1', index=False)
            # print(path)
            if input.L2():
                csv2 = pd.read_csv(f'{path}/L2_train_data.csv')
                csv2.to_excel(writer, sheet_name='Sheet2', index=False)
            if input.L3_choices():
                csv3 = pd.read_csv(f'{path}/L3_train_data.csv')
                csv3.to_excel(writer, sheet_name='Sheet3', index=False)
            writer.save() # type: ignore            
            return f'{path}/all_train_data.xlsx'
        
        return _()
        
    @session.download(
        filename=lambda: f"data-{date.today().isoformat()}-{np.random.randint(100,999)}.csv"
    )
    def dl_result_test():
        def _():
            # 将三层的数据放到三张sheet上
            path = str(Path(__file__).parent/"model_checkpoint")
            # print(path)
            csv1 = pd.read_csv(f'{path}/L1_test_data.csv')
            
            
            # 创建一个Excel写入器，用于将数据保存为一个带有三个sheet的CSV文件
            writer = pd.ExcelWriter(f'{path}/all_test_data.xlsx')
            # 将每个CSV数据写入不同的sheet
            csv1.to_excel(writer, sheet_name='Sheet1', index=False)
            if input.L2():
                csv2 = pd.read_csv(f'{path}/L2_test_data.csv')
                csv2.to_excel(writer, sheet_name='Sheet2', index=False)
            
            if input.L3_choices():
                csv3 = pd.read_csv(f'{path}/L3_test_data.csv')
                csv3.to_excel(writer, sheet_name='Sheet3', index=False)
            writer.save() # type: ignore 
            return f'{path}/all_test_data.xlsx'
        return _()

    # ====
    
    # 训练结束后
    @reactive.Calc
    def train_metric_table():
        model_save_files = os.getcwd()+f"/model_checkpoint_{_number}"
        # print(model_save_files)
        df = pd.DataFrame([])
        # print(df)
        
        train_data = pd.read_csv(model_save_files+"/L1_train_data.csv")
        df = utils.cal_df(train_data)
        
        if input.L2():
            train_data = pd.read_csv(model_save_files+"/L2_train_data.csv")
            df = pd.concat([df , utils.cal_df(train_data)], axis=1)
        if input.L3_choices():
            train_data = pd.read_csv(model_save_files+"/L3_train_data.csv")
            df = pd.concat([df , utils.cal_df(train_data)], axis=1)
        # 修改输出结果样式
        # print(df)
        df.to_csv(f'model_checkpoint_{_number}/train_metric.csv')
        
    @reactive.Calc
    def test_metric_table():
        model_save_files = os.getcwd()+f"/model_checkpoint_{_number}"
        df = pd.DataFrame([])
        # print(df)
        
        test_data = pd.read_csv(model_save_files+"/L1_test_data.csv")
        df = utils.cal_df(test_data)
        if input.L2():
            test_data = pd.read_csv(model_save_files+"/L2_test_data.csv")
            df = pd.concat([df , utils.cal_df(test_data)], axis=1)
        if input.L3_choices():
            test_data = pd.read_csv(model_save_files+"/L3_test_data.csv")
            df = pd.concat([df , utils.cal_df(test_data)], axis=1)
        df.to_csv(f'model_checkpoint_{_number}/test_metric.csv')
        
    # 展示结果
    @ output
    @ render.table(index=True)
    @reactive.event(input.train_plot)
    def show_train_metric_table():
        df = pd.read_csv(f'model_checkpoint_{_number}/train_metric.csv',index_col=0)
        return df
        
        
        
    @ output
    @ render.table(index=True)
    @reactive.event(input.test_plot)
    def show_test_metric_table():
        # a = _trained()
        df = pd.read_csv(f'model_checkpoint_{_number}/test_metric.csv',index_col=0)
        return df
        
    # ====
    # 打包模型
    # 将model_checkpoint压缩成一个zip，用户自行下载保存，未来上传到服务器进行预测
    def compress_to_zip(source_folder, output_zip_file):
        """
        将一个文件夹中的文件打包成zip文件

        参数：
        source_folder (str): 要打包的文件夹路径
        output_zip_file (str): 输出的zip文件路径

        返回：
        无
        """
        # 获取文件总数
        total_files = 0
        for root, _, files in os.walk(source_folder):
            total_files += len(files)
        
            
        # 创建进度条
        with ui.Progress(min=1, max=total_files) as p:
            with zipfile.ZipFile(output_zip_file, 'w') as zipf:
                i = 1
                for root, _, files in os.walk(source_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, source_folder))
                        # 更新进度条
                        p.set(i, message="zipping_models")
                        i+=1
    
        
        
    def model_zipper(file_path=Path(__file__).parent/"model_checkpoint",endings=None):
        """模型打包函数，返回zip文件位置

        Args:
            file_path (_type_, optional): _description_. Defaults to Path(__file__).parent/"model_checkpoint".
            endings (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        output_zip_file = f'{file_path}_{endings}.zip'
        
        return output_zip_file
    
    @session.download()
    def model_zip_download():
        # 下载完成训练的模型，先打包
        source_folder = Path(__file__).parent/f"model_checkpoint_{_number}"
        output_zip_file = f'model_checkpoint_{_number}.zip'
        compress_to_zip(source_folder,output_zip_file)
        return output_zip_file
    
    # ====
    # 预测步骤
    # 若是用户专属模型则先下载，解压。若是默认模型则直接使用model_checkpoint中的数据
    
    @reactive.Calc
    def get_users_model():
        file = input.users_model()
        zip_path =file[0]['datapath'] # 获取原zip路径
        # 解压缩目标目录
        extract_to = Path(__file__).parent/"model_checkpoint_users"
        print(zip_path)
        
        # 打开 ZIP 文件
        # 获取 ZIP 文件中的文件数量
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            num_files = len(zip_ref.infolist())
        print(num_files)
        # print(num_files)
        with ui.Progress(min= 1,max=num_files) as p:
            i=1
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    # 解压缩单个文件
                    zip_ref.extract(file_info, extract_to)
                    p.set(i)  # 更新进度条
                    i+=1
        
        print('解压缩完成')
        return '上传完成'
    
    @reactive.Calc
    def get_params():
        """读取模型结构
        
        Return: 
            [num_Layer,lis]:模型层数
        """
        
        def read_json_file(file_path) -> dict | None:
            """在模型储存文件中查询并且读取模型结构json文件

            Args:
                file_path (_type_): _description_

            Returns:
                dict: _description_
            """
            # 在文件夹中检索一个JSON文件是否存在，如果存在则读取，否则返回None：
            if os.path.exists(file_path):  # 检查文件是否存在
                with open(file_path, "r") as json_file:
                    data = json.load(json_file)  # 读取JSON数据
                return data
            else:
                return None
        def get_model_sequential(dic,model_path) -> utils.model_sequential:
            """从结构json文件读取的字典中，生成meta模型结构

            Args:
                dic (dict): 字典
                dic ={
                        'id':{
                            'Model Categories': str (meta_xxx),
                            'paraments':list | None
                        }
                    }
            Returns:
                utils.model_sequential: _description_
            """
            lis = []
            meta_models = lazy_import('models.meta_models')
            for i, model_id in  enumerate(dic.keys()) :
                _d = dic[model_id]
                model_name = _d['Model Categories']
                _model = getattr(meta_models,model_name)
                model = _model()
                if "paraments" in _d.keys():
                    setattr(model,'ECFP_Params',_d["paraments"])
                    # print(_d["paraments"])
                setattr(model,'model_save_files',model_path)
                setattr(model,'id' ,i)
                lis.append(model)
                # print(model.name)
            return utils.model_sequential(*lis)
        
        model_path = Path(__file__).parent/"model_checkpoint"
        # "a":"上传模型","b":"默认模型(口服急性)"
        if input.model_check()=="a":
            model_path = Path(__file__).parent/"model_checkpoint_users"
        
        print(model_path)
        
        num_Layer = 1
        lis = []
        # 第一层模型读取
        model_structure_L1 = read_json_file(file_path=str(model_path)+'/model_structure.json')
        print(model_path)
        print(model_structure_L1)
        
        if type(model_structure_L1) == dict:
            model_sequential_L1 = get_model_sequential(dic = model_structure_L1,model_path=model_path )
            lis.append(model_sequential_L1)
        else:
            print("There has some problem in model_checkpoint cannot load model_sequential")
            num_Layer-=1
        
        # 实现在预测加载用户模型时，读取结构。识别层数，修改L3储存形式为json
        model_structure_L2 = read_json_file(file_path=str(model_path)+'/model_structure_L2.json')
        if type(model_structure_L2) == dict:
            model_sequential_L2 = get_model_sequential(dic = model_structure_L2, model_path=model_path)
            lis.append(model_sequential_L2)
            num_Layer+=1
        else:
            pass

        model_structure_L3 = read_json_file(file_path=str(model_path)+'/model_structure_L3.json')
        if type(model_structure_L3) == dict:
            lis.append(model_structure_L3)
            num_Layer+=1
        else:
            pass
        return [num_Layer,lis]
    
    @output
    @render.text
    @reactive.event(input.zipping)
    def finish_zip():
        """返回模型结构
        """
        _f = get_users_model()

        model_info = get_params()
        # print(model_info)
        num_layer = model_info[0]
        model_lis = model_info[1]
        _str = ''
        if num_layer ==0:
            return "There has some problem in model_file."
        elif num_layer ==1 or num_layer==2:
            x ='layers'
            if num_layer==1:
                x='layer' 
                
            _str = f'The model has {num_layer} {x}\n'
            for _m in model_lis:
                _str = _str + getattr(_m,'get_models_mark')()
            return _str
        elif num_layer == 3:
            _str = 'The model has 3 layers'
            for _m in model_lis[:2]:
                _str = _str + getattr(_m,'get_models_mark')()
            _str +='Non-negative Least Squares for the third layer'
            return _str
            
        return _str

    # =====
    # 预测步骤2
    # 展示样例数据
    
    @output
    @render.table(index=True)
    def sample_test():
        exemple_df2 = pd.read_csv(Path(__file__).parent/"www/files/sample2.csv")
        # print(exemple_df2.shape)
        return exemple_df2
    
    @session.download()
    def download_exemple_predf():
        return str(Path(__file__).parent/"www/files/sample2.csv")
    
    
    #获取预测数据
    @reactive.Calc
    def get_test_df():
        files_info = input.predict_set()
        df = pd.read_csv(files_info[0]["datapath"])
        return df
    
    # =====
    # 开始预测，先根据模型类型选择模型路径，再进行预测
    
    @reactive.Calc
    @reactive.event(input.do_predicting)
    def get_pre_model():
        # 选择模型路径model_path
        # "a":"上传模型","b":"默认模型(口服急性)"
        print("===========================")
        model_path = Path(__file__).parent/"model_checkpoint_default"
        if input.model_check()=="a":
            model_path = Path(__file__).parent/"model_checkpoint_users"
            print(model_path)
        LAY = lazy_import('models.LAY2')
        # 获取预测数据
        with ui.Progress(min=1, max=3 ) as p:
            p.set(message="Calculation in progress", detail="This may take a while...")
            test_df = get_test_df()
            p.set(1, message="Level 1...")
            eb_model = LAY.ensemble_models_new(model_save_files=str(model_path))
            eb_model.load_pretrained_metamodels(test_df)
            p.set(2, message="Level 2...")
            eb_model.load_pretrained_L2models()
            p.set(3, message="Level 3...")
            eb_model.load_pretrained_L3model(test_df)
        return eb_model
    
    @output
    @render.text
    def pre_finish_text():
        model = get_pre_model()
        
        if model ==None:
            return None
        return "完成!!"

    # 获取预测结果
    @session.download()
    def predict_result_download():
        model = get_pre_model()
        df = model.predict_df    
        _path = Path(__file__).parent /"www/files/predict_data.csv"
        # print(type(df))
        model.L2_predict_df.to_csv(Path(__file__).parent /"www/files/predict_data_L2.csv") 
        df.to_csv(_path)    
        return str(_path)
www_dir = Path(__file__).parent / "www"
app = App(app_ui, server, static_assets=www_dir)