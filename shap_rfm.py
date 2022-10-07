#import 相依套件
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st
import seaborn as sns
import csv

from streamlit_shap import st_shap
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

st.set_option('deprecation.showPyplotGlobalUse', False)
#設置標題和上傳資料
st.title('SHAP values and RFM analysis')
uploaded_file = st.file_uploader("Choose a file")
#如上傳資料不為空值 load data
if uploaded_file is not None:
  data = pd.read_csv(uploaded_file,encoding= 'unicode_escape')
  # 側邊欄位sidebar
  tar_columns = st.sidebar.multiselect('target column', data.columns)
  index_ = st.sidebar.multiselect('positive(1) and negative(0) ', [0,1])
  R_columns = st.sidebar.multiselect('Recency column',data.columns)
  F_columns = st.sidebar.multiselect('Frequency columns', data.columns)
  M_columns = st.sidebar.multiselect('Monetary columns', data.columns)
  if (len(R_columns) and len(F_columns) and len(M_columns))!= 0:
    F_data = data[F_columns[0]]
    R_data = data[R_columns[0]]
    M_data = data[M_columns[0]]
  # 可依RFM的重要程度排序
  RFM_ord =['R>F>M','R>M>F','M>R>F','M>F>R','F>M>R','F>R>M']
  print(RFM_ord[0].split('>')[0:2])
  RFM_ord_columns = st.sidebar.multiselect('ordered RFM', RFM_ord)
  st.write(data)
  st.success('Load data success !')
  # target columns 不為空值
  if len(tar_columns) != 0:
    tar_data = data[tar_columns[0]]
    mean =np.array(tar_data.values).mean()
    for i in range(len(tar_data)):
      tar_ = tar_data.values.ravel()[i]
      if tar_ <= mean:
        tar_data[i] = 0
      if tar_ > mean:
        tar_data[i] = 1
      # columns欄位值 => 1
    if st.sidebar.button('Click!!'):
      Y = (tar_data ==1)  # Convert from string "Yes"/"No" to binary
      feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.float64,int,float]]
      X = data[feature_names]
      # 分成訓練集/測試集
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
      # Build the model with the random forest regression algorithm:
      model = RandomForestClassifier(max_depth=6, random_state=0, n_estimators=10)
      model.fit(X_train, Y_train)

      # # 預測第五列(columns)
      row_to_show = 5
      data_for_prediction = X_test.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
      data_for_prediction_array = data_for_prediction.values
      # Create object that can calculate shap values
      explainer = shap.TreeExplainer(model)

      # Calculate Shap values
      # shap解釋預測第五列(columns)
      #shap_values = explainer.shap_values(X)
      shap_values = explainer.shap_values(data_for_prediction)
      # shap解釋所有的測試集
      shap_values2 = explainer.shap_values(X_test)
      #rf_shap_values = shap.KernelExplainer(model.predict, val_X)
      # 畫shap value圖,positive（1） and negative(0)
      if len(index_) != 0:
        t = index_[0]
        st_shap(shap.force_plot(explainer.expected_value[t], shap_values[t], data_for_prediction))
        st_shap(shap.summary_plot(shap_values2[t],X_test))
        #根據值將資料分級
        #col_val: 輸入值, level: 分級級距
      if len(R_columns)!=0 and len(F_columns)!=0 and len(M_columns)!=0:
          #分級
          # 根據值將資料分級
          # col_val: 輸入值, level: 分級級距
          def get_level(col_val, level: list):
            for idx, val in enumerate(reversed(level)):
              if col_val >= val:
                return f'≥{int(val)}'
          recency_level = [0, R_data.median()]
          recency_label = [f'≥{int(i)}' for i in recency_level]
          data['recency_label'] = R_data.apply(lambda x: get_level(x, recency_level))

          frequency_level = [0, F_data.median()]
          frequency_label = [f'≥{int(i)}' for i in frequency_level]
          data['frequency_label'] = F_data.apply(lambda x: get_level(x, frequency_level))

          monetary_level = [0, M_data.median()]
          monetary_label = [f'≥{int(i)}' for i in monetary_level]
          data['monetary_label'] = M_data.apply(lambda x: get_level(x, monetary_level))
          data['user'] = 1
        # 利用 seaborn 繪圖，分成四個區塊為各個客戶群，裡面兩條長條圖分別為高消費力與低消費力的人數
        # RFM分析,取前兩個值ex:R、M做分析
        # C3取2所以有3種呈現圖
          if RFM_ord[0] == 'R>F>M' or RFM_ord[0] =='F>R>M':
            # 利用 seaborn 繪圖，分成四個區塊為各個客戶群，裡面兩條長條圖分別為高消費力與低消費力的人數
            g = sns.FacetGrid(
              data,  # 來源資料表
              col="recency_label",  # X資料來源欄位
              row="frequency_label",  # Y資料來源欄位
              col_order=recency_label,  # X資料順序
              row_order=frequency_label[::-1],  # Y資料順序
              margin_titles=True)  # 小圖啟用

            g = g.map_dataframe(
              sns.barplot,  # 資料顯示的模式
              x='monetary_label',  # 小圖表X資料來源
              y='user',  # 小圖表Y資料來源，訪客計數
              order=monetary_label,  # X資料順序
              estimator=sum,  # Y計數加總
              palette=sns.color_palette("muted"))  # 畫布色調
            g = g.set_axis_labels('Recency', 'Frequency').add_legend()
            st.pyplot()

          if RFM_ord[0] == 'R>M>F' or RFM_ord[0] == 'M>R>F':
            g = sns.FacetGrid(
              data,  # 來源資料表
              col="recency_label",  # X資料來源欄位
              row="monetary_label",  # Y資料來源欄位
              col_order=recency_label,  # X資料順序
              row_order=monetary_label[::-1],  # Y資料順序
              margin_titles=True)  # 小圖啟用
            g = g.map_dataframe(
              sns.barplot,  # 資料顯示的模式
              x='frequency_label',  # 小圖表X資料來源
              y='user',  # 小圖表Y資料來源，訪客計數
              order=frequency_label,  # X資料順序
              estimator=sum,  # Y計數加總
              palette=sns.color_palette("muted"))  # 畫布色調
            g = g.set_axis_labels('Recency', 'Monetary').add_legend()
            st.pyplot()

          if RFM_ord[0] == 'M>F>R' or RFM_ord[0] == 'F>M>R':
            g = sns.FacetGrid(
              data,  # 來源資料表
              col="monetary_label",  # X資料來源欄位
              row="frequency_label",  # Y資料來源欄位
              col_order=monetary_label,  # X資料順序
              row_order=frequency_label[::-1],  # Y資料順序
              margin_titles=True)  # 小圖啟用
            g = g.map_dataframe(
              sns.barplot,  # 資料顯示的模式
              x='recency_label',  # 小圖表X資料來源
              y='user',  # 小圖表Y資料來源，訪客計數
              order=recency_label,  # X資料順序
              estimator=sum,  # Y計數加總
              palette=sns.color_palette("muted"))  # 畫布色調
            g = g.set_axis_labels('monetary', 'Frequency').add_legend()
            st.pyplot()
          vip = data[
           (data['frequency_label'] == frequency_label[-1]) &
            (data['recency_label'] == recency_label[-1]) &
           (data['monetary_label'] == monetary_label[-1])]
          st.dataframe(vip)
          @st.cache
          def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
          csv = convert_df(vip)
          st.download_button(
            "Press to Download",
            csv,
            "vip_dataframes.csv",
            "text/csv",
            key='download-csv')