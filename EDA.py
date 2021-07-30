import pandas as pd 
import numpy as np
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import sweetviz as sv
import os
import webbrowser #for displaying pandas-profiling report on a separate page
from pandas_profiling import ProfileReport
# the crux of UI integration is that sometimes you have to have a bridge of sorts, 
# helper methods or drop down to a level where both libraries can talk to each other, 
# in this case a tkinter canvas.
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from autoviz.AutoViz_Class import AutoViz_Class
from lightgbm import LGBMClassifier as lgbmc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def handle_tools():
    path_ht = values['-IN-']
    df_ht = pd.read_csv(path_ht)

    if values['-TOOL-']=='sweetviz':
        report = sv.analyze(df_ht)
        report.show_html("sweet_report.html")

    if values['-TOOL-']=='autoviz':
        AV = AutoViz_Class()
        av = AV.AutoViz(path_ht,verbose=1)

    if values['-TOOL-']=='pandas profiling':
        design_report = ProfileReport(df_ht)
        design_report.to_file(output_file='report.html')
        webbrowser.open('file://' + os.path.realpath('report.html'))

    # Autoviz library issues to be resolved

def viz_window(values):
    path_vw = values['-IN-']
    df_vw = pd.read_csv(path_vw)
    df_vw = df_vw.select_dtypes([np.number]) # removing non-numeric columns
    df_cleaned = df_vw.dropna()
    df_max_scaled = df_cleaned.copy()
    for column in df_vw.columns: # normalising the data (Min-Max Scaling)
        df_max_scaled[column] = (df_max_scaled[column] - 
        df_max_scaled[column].min())/(df_max_scaled[column].max() - 
        df_max_scaled[column].min())
    df1 = df_max_scaled # changed name to df1

    all_columns = list(df1.columns)
    plot_list = ['scatter','line']
    layout = [
        [
            sg.Text("X axis: "), # command-slash 
            sg.Combo(all_columns,default_value=all_columns[0],
            enable_events=True,key="-X-")
        ],
        [
            sg.Text("Y axis: "),
            # sg.Combo(all_columns,default_value=all_columns[0],
            # enable_events=True,key="-Y-")
            sg.Listbox(all_columns,default_values=[all_columns[0]],
            enable_events=True,select_mode='multiple',size=(25,1),key="-Y-")
        ],
        [
            sg.Text("Plot: "),
            sg.Combo(plot_list,default_value=plot_list[0],enable_events=True,
            size=(25,1),key="-PLOT-")
        ],
        [
            sg.Button("Show",enable_events=True,key="-SHOW_PLT-")
        ]
    ]
    window = sg.Window("Visualization",layout,size=(650,200))
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Exit"):
            break
        if event == "-SHOW_PLT-":
            handle_plot(values,df1)
        #adds visualisation functionality with matplotlib
        
    window.close()


def handle_plot(values,df1):
    _VARS = {'window': False}
    layout = [
        [sg.Canvas(key='figCanvas')],
        [sg.Button('Exit')]
        ]
    _VARS['window'] = sg.Window('Graph Window',layout,finalize=True,resizable=True,
    element_justification="right")

    list_val = list(values["-Y-"]) # to plot multiple y axes
    if values["-PLOT-"] == 'scatter':
        f, ax = plt.subplots(figsize=(15,15))
        plt.title('Scatter Plot')
        ax2 = ax.twinx()
        for i in range(len(list_val)):
            ax.scatter(df1[values["-X-"]], df1[list_val[i]],label=list_val[i])
        ax.set_xlabel(values["-X-"])
        ax.legend(loc='upper right')
        # Instead of plt.show
        draw_figure(_VARS['window']['figCanvas'].TKCanvas, f)

    elif values["-PLOT-"] == 'line':
        f, ax = plt.subplots(figsize=(20,15))
        plt.title('Line Plot')
        ax2 = ax.twinx()
        for i in range(len(list_val)):
            ax.plot(df1[values["-X-"]], df1[list_val[i]],label=list_val[i])
        ax.set_xlabel(values["-X-"])
        ax.legend(loc='upper right')
         # Instead of plt.show
        draw_figure(_VARS['window']['figCanvas'].TKCanvas, f)

    while True:
        event, values = _VARS['window'].read(timeout=200)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
    
    _VARS['window'].close()



def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg



def handle_model():
    model_list = ["Regression","Classification"]
    target_list = list(df.columns)
    layout = [
        [
            sg.Text("Choose Model"),
            sg.Combo(model_list,default_value=model_list[0],size=(25,1),
            key="-MODEL_FIT-")
        ],
        [
            sg.Text("Choose Features"),
            sg.Listbox(target_list,enable_events=True,select_mode='multiple',
            size=(25,1),key="-FEATURE-")
        ],
        [
            sg.Text("Choose Target Variable"),
            sg.Combo(target_list,default_value=target_list[0],size=(25,1),
            key='-TARGET-')
        ],
        [
            sg.Button("Fit",enable_events=True,size=(25,1),key="-FIT-")
        ],
        [
            sg.Text("", size=(50,1),key='-prediction-', pad=(5,5), font='Helvetica 12')
        ],
        [
            sg.ProgressBar(50, orientation='h', size=(100,10), key='progressbar')
        ],
        [sg.Text(" ")],
        [
            sg.Button("To Predict",enable_events=True,size=(25,1),key="-PREDICT-")
        ]
    ]
    window = sg.Window("Model Fit Window",layout)
    progress_bar = window['progressbar']
    prediction_text = window['-prediction-']
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED,"Exit"):
            break
        if event == "-FIT-":
            prediction_text.update("Fitting model...")
            for i in range(50):
                event, values = window.read(timeout=10)
                progress_bar.UpdateBar(i + 1)
            mod,score = sklearn_model(values)
            prediction_text.update("Accuracy of Model is: {}%".format(score*100))
        if event == "-PREDICT-":
            sklearn_predict(mod,values)
    window.close()
        #To add prediction functionalities 


def sklearn_model(values):
    le = LabelEncoder() # to later check for encoded columns
    for i,column in enumerate(df.columns):
        if (type(df[column][0]) == str): # Encoding non-numeric columns with LabelEncoder
            le = LabelEncoder()
            cat_arr = df.iloc[:,i].values
            cat_list = le.fit_transform(cat_arr)
            df[column] = cat_list
    
    X = df[list(values["-FEATURE-"])]
    y = df[values["-TARGET-"]]
    if values["-MODEL_FIT-"] == "Classification":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=20,max_depth=4)
        clf.fit(X,y)
        return clf, np.round(clf.score(X,y),3)
    if values["-MODEL_FIT-"] == "Regression":
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X,y)
        return regressor, np.round(regressor.score(X,y),3)

def sklearn_predict(mod,values):
    input_col = []
    feature_col = values["-FEATURE-"]
    cols = len(feature_col)
    pred_list = []
    for i in values["-FEATURE-"]:
        input_col.append([sg.Text("{}".format(i)),sg.Input(key='{}'.format(i))])
    layout = [
        [sg.Text("Enter values to predict")],
        *input_col,
        [sg.Button("Predict",enable_events=True,size=(25,1),key="-DISP-")],
        [sg.Text("",size=(50,1),key="-OUTPUT-")]
    ]
    window = sg.Window("Prediction Window",layout)
    result_text = window["-OUTPUT-"]
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED,"Exit"):
            break
        if event == "-DISP-":
            # for column in feature_col: # normalising the data (Min-Max Scaling)
            #     values[column] = ((float(values[column]) - df_max_scaled[column].min())/(df_max_scaled[column].max() - df_max_scaled[column].min()))
            for i in feature_col:
                pred_list.append(float(values[i]))
            arr = np.array(pred_list)
            arr_2d = np.reshape(arr, (1,cols))
            predicted = mod.predict(arr_2d)
            result_text.update("Predicted result is {}".format(predicted))
            pred_list = []

# def handle_pycaret(values):
#     if values["-MODEL_FIT-"] == "Classification":
#         from pycaret.classification import setup,create_model,compare_models,plot_model
#         clf1 = setup(df, target = values["-TARGET-"], session_id=786)
#         extracted_model = compare_models()
#         model = create_model(extracted_model)
#         plot_model(model)

#     elif values["-MODEL_FIT-"] == "Regression":
#         from pycaret.regression import setup,create_model,compare_models,plot_model
#         reg1 = setup(df, target = values["-TARGET-"], session_id=786)
#         extracted_model = compare_models()
#         model = create_model(extracted_model)
#         plot_model(model)

def show_table():
    df_st = df.dropna()
    data = df_st.values.tolist()
    header_list = list(df.columns)


    layout = [
        [
            sg.Table(values=data,
                  headings=header_list,
                  font='Helvetica',
                  pad=(25,25),
                  display_row_numbers=False,
                  auto_size_columns=True,
                  num_rows=min(25, len(data))) 
        ]
    ]
    window = sg.Window("Dataset",layout,size=(1000,350))
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
    window.close()

file_choose = [
    [
        sg.Text("Data File"),
        sg.In(size=(25,1),enable_events=True,key="-FILE-"),
        sg.FileBrowse(key="-IN-")
    ],
    [
        sg.Button("Show Data",enable_events=True,key="-SHOW-")
    ]
]

tool_list = ['sweetviz','autoviz','pandas profiling']
tool_choose = [
    [sg.Text("   ")],
    [
        sg.Text("Choose Tool"),
        sg.Combo(tool_list,default_value='sweetviz',size=(25,1),
        key="-TOOL-")
    ],
    [sg.Button("Perform EDA",enable_events=True,key="-EDA-")]
]

plot_choose = [
    [sg.Text("   ")],
    [sg.Text("To Visualize:")],
    [sg.Button("Visualisation options",enable_events=True,
    key="-VIZ-")]

        # sg.Text("X axis"), # command-slash 
        # sg.Combo(all_columns,default_value=all_columns[0],
        # enable_events=True,key="-X-"),
]


model_choose = [
    [sg.Text("   ")],
    [sg.Text("To Fit a Model:")],
    [sg.Button("Click here",enable_events=True,size=(25,1),key="-MODEL-")]
]


layout = [
    file_choose,tool_choose,plot_choose,model_choose
]

window = sg.Window("EDA Report",layout,size=(750,350))
while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED,'Exit'):
        break
    if event == '-FILE-':
        path = values['-IN-']
        df = pd.read_csv(path)
        # df = df.select_dtypes([np.number]) # removing non-numeric columns
        # df_cleaned = df.dropna()
        # df_max_scaled = df_cleaned.copy()
        # for column in df.columns: # normalising the data (Min-Max Scaling)
        #     df_max_scaled[column] = (df_max_scaled[column] - 
        #     df_max_scaled[column].min())/(df_max_scaled[column].max() - 
        #     df_max_scaled[column].min())
        # df = df_max_scaled
    if event == '-EDA-':
        handle_tools()
    if event == '-VIZ-':
        viz_window(values)
    if event == '-MODEL-':
        handle_model()
    if event == '-SHOW-':
        show_table()

window.close()