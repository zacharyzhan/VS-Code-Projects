import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 读取文本文件
with open(r"C:\Users\zacha\Documents\WorkSpace\VS Code Projects\Data Mining\sy1\北斗介绍.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 使用jieba进行分词
words = jieba.cut(text)
word_list = " ".join(words)

# 创建词云对象
wc = WordCloud(
    font_path="simsun.ttc",  
    width=800,
    height=600,
    background_color="white",
    max_words=200,
    max_font_size=100,
)

# 生成词云图
wc.generate(word_list)

# 显示词云图
plt.imshow(wc)
plt.axis("off")
plt.show()

# 保存词云图
wc.to_file(r"C:\Users\zacha\Documents\WorkSpace\VS Code Projects\Data Mining\sy1\北斗导航系统词云图.png")
