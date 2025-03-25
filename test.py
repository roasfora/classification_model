import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dicionário com as palavras de cada tópico
topics_words = {
    "AI & Technology Concepts": ["make", "artificial", "computer", "data", "think", "intelligence", "ai"],
    "AI & Society": ["time", "change", "thing", "think", "know", "dont", "world", "like", "people", "ai"],
    "AI in Social Media & Engagement": ["facewithopenmouth", "great", "rollingonthefloorlaughing", "thank", "sir", "make", "im", "ai", "redheart", "video"],
    "Ethical & Philosophical Debates": ["smilingfacewithsmilingeyes", "look", "hai", "real", "movie", "na", "like", "grinningfacewithsweat", "robot", "facewithtearsofjoy"],
    "AI & Automation Concerns": ["think", "dont", "like", "humanity", "world", "life", "need", "people", "human", "ai"]
}

# Dicionário com os pesos das palavras por tópico (exemplo)
topic_weights = {
    "AI & Technology Concepts": {
        "make": 5590.240634495856,
        "artificial": 1.6416255243480389,
        "computer": 571.641928159537,
        "data": 361.8902851071556,
        "think": 5608.990967009148,
        "intelligence": 37.5993479603201,
        "ai": 50289.94948131429
    },
    "AI & Society": {
        "time": 2836.541889554349,
        "change": 12.430521417300358,
        "thing": 5542.543366579497,
        "think": 5933.6429600654,
        "know": 4481.541428945999,
        "dont": 4469.013304070758,
        "world": 1160.5446770619776,
        "like": 8051.961589659473,
        "people": 2289.844519840016,
        "ai": 38311.43329978392
    },
    "AI in Social Media & Engagement": {
        "facewithopenmouth": 0.3642046721011818,
        "great": 1928.0380905818736,
        "rollingonthefloorlaughing": 0.7798662856372516,
        "thank": 186.1891042574703,
        "sir": 0.2005557674889377,
        "make": 4253.580319930161,
        "im": 10479.998849009124,
        "ai": 11766.86753666181,
        "redheart": 0.2004594295189207,
        "video": 7050.503710462261
    },
    "Ethical & Philosophical Debates": {
        "smilingfacewithsmilingeyes": 0.9769891729103164,
        "look": 198.1340590530888,
        "hai": 0.2001096128270747,
        "real": 151.8000400128445,
        "movie": 0.2016506222113869,
        "na": 0.2012390253014064,
        "like": 1577.8627889900197,
        "grinningfacewithsweat": 0.2007191344136209,
        "robot": 4.7204610676029874,
        "facewithtearsofjoy": 0.2005334479494121
    },
    "AI & Automation Concerns": {
        "think": 29.13511166595135,
        "dont": 1.1583702900373023,
        "like": 953.7491792859582,
        "humanity": 0.2009998980130556,
        "world": 19.456543534829827,
        "life": 17.211787073779092,
        "need": 36.78826630396493,
        "people": 0.8557113635268787,
        "human": 4.797703952023225,
        "ai": 2101.381193099661
    }
}

# Criar um DataFrame apenas com as palavras listadas
df_filtered = pd.DataFrame()
for topic, words in topics_words.items():
    for word in words:
        if word in topic_weights[topic]:
            df_filtered.loc[word, topic] = topic_weights[topic][word]

# Normalizar os pesos para o intervalo [0, 1]
df_normalized = df_filtered / df_filtered.max().max()

plt.figure(figsize=(12, 8))
sns.heatmap(df_normalized, annot=False, cmap="YlOrRd", linewidths=0.5)
plt.title("Topic-Term Weights Heatmap", pad=20)
plt.xlabel("Topic ", labelpad=10)
plt.ylabel("Words", labelpad=10)
plt.xticks(rotation=45, ha='right', fontsize=8)  # Tamanho da fonte do eixo X
plt.yticks(fontsize=8)  # Tamanho da fonte do eixo Y
plt.tight_layout()
plt.show()