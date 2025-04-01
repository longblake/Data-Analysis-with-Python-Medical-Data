
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = df['weight'].apply(lambda x: 1 if x>25 else 0)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x<=1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x<=1 else 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat = df_cat.rename(columns={'value': 'category'})
    df_cat = df_cat.groupby(['cardio', 'variable', 'category']).size().reset_index(name='count')

    # 7 & 8
    fig = sns.catplot(x='variable', y='count', hue='category', col='cardio', kind='bar', data=df_cat)

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df_heat[
    (df_heat['ap_lo'] <= df_heat['ap_hi']) &
    (df_heat['height'] >= df_heat['height'].quantile(0.025)) &
    (df_heat['height'] <= df_heat['height'].quantile(0.975)) &
    (df_heat['weight'] >= df_heat['weight'].quantile(0.025)) &
    (df_heat['weight'] <= df_heat['weight'].quantile(0.975))
    ]
    # 12
    corr = None

    # 13
    mask = None



    # 14

    #fig, ax = None

    # 15



    # 16

    #fig.savefig('heatmap.png')
    #return fig

