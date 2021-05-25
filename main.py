import numpy as np
import pandas as pd
import seaborn as sns
import pandas_profiling as pp
from sklearn import svm
from sklearn import model_selection
from pyecharts.charts import Pie,Bar
from pyecharts import options as opts
data=pd.read_csv(r'winequality-red.csv')
"""quality_num=pd.DataFrame(data.value_counts('quality'))
quality_num=quality_num.sort_values('quality')
quality_num.to_csv('quality_num.csv')
quality_num=pd.read_csv('quality_num.csv')
x=quality_num['quality']
y=quality_num['0']
x_data = x
y_data = y

c = (
    Bar()
    .add_xaxis(list(x_data))
    .add_yaxis("", list(y_data), stack="stack1")
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .render("品质分布.html")
)"""

report=pp.ProfileReport(data)
report.to_file('report.html')