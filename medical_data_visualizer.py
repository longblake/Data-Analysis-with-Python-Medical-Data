
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

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
    g = sns.catplot(x='variable', y='count', hue='category', col='cardio', kind='bar', data=df_cat)
    g.set_axis_labels("variable", "total")
    # 9
    g.savefig('catplot.png')
    return g.fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
    (df['ap_lo'] <= df['ap_hi']) &
    (df['height'] >= df['height'].quantile(0.025)) &
    (df['height'] <= df['height'].quantile(0.975)) &
    (df['weight'] >= df['weight'].quantile(0.025)) &
    (df['weight'] <= df['weight'].quantile(0.975))
    ]
    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # 14
    fig, ax = plt.subplots(figsize=(16, 9))

    # 15
    sns.heatmap(corr, mask=mask, square=True, annot=True, linewidths=0.5, fmt="0.1f")

    # 16
    fig.savefig('heatmap.png')
    return fig

