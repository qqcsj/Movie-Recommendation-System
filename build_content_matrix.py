# --- 这是 build_content_matrix.py (V8 智能版) ---
import os
import django
import pickle
import jieba

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "movie.settings")
django.setup()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from user.models import Movie


def build_and_save_matrix():
    print("开始获取所有电影...")
    all_movies = list(Movie.objects.all())

    corpus = []
    print("正在构建智能语料库...")
    for movie in all_movies:
        tags_str = ' '.join([tag.name for tag in movie.tags.all()])
        director_str = movie.director if movie.director else ""
        intro_str = movie.intro if movie.intro else ""
        name_str = movie.name if movie.name else ""
        leader_str = movie.leader if movie.leader else ""
        # (我们把所有文本信息合并，用于 TF-IDF 搜索)
        content = f"{name_str} {tags_str} {director_str} {leader_str} {intro_str}"
        # ⭐️ 2. 【V9 中文分词修复】 ⭐️
        #    (不是直接添加 content, 而是添加 "分词后" 的 content)
        tokenized_content = ' '.join(jieba.cut_for_search(content))

        corpus.append(tokenized_content)

    print(f"语料库构建完毕，总计 {len(corpus)} 部电影。")

    # ⭐️【修复“千寻”问题】⭐️
    # min_df=2 (忽略出现少于2次的词) -> min_df=1 (包含所有词)
    tfidf = TfidfVectorizer(min_df=1)

    tfidf_matrix = tfidf.fit_transform(corpus)
    print(f"TF-IDF 向量计算完毕，矩阵形状: {tfidf_matrix.shape}")

    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"相似度矩阵计算完毕，矩阵形状: {cosine_sim_matrix.shape}")

    # --- 9. 创建“地图” ---
    movie_id_map = {movie.id: index for index, movie in enumerate(all_movies)}
    print("ID 地图创建完毕。")

    # --- 10. 保存所有“大脑”文件 (V7 版 3文件) ---
    print("正在保存“AI 翻译官” (tfidf_vec.pkl)...")
    with open('tfidf_vec.pkl', 'wb') as f_vec:
        pickle.dump(tfidf, f_vec)

    print("正在保存“电影特征矩阵” (tfidf_matrix.pkl)...")
    with open('tfidf_matrix.pkl', 'wb') as f_matrix:
        pickle.dump(tfidf_matrix, f_matrix)

    print("正在打包 (矩阵, 地图) ...")
    data_to_save = (cosine_sim_matrix, movie_id_map)  # 这是一个元组 (Tuple)

    print("正在保存“内容相似度”大脑 (content_sim.pkl)...")
    with open('content_sim.pkl', 'wb') as f_sim:
        pickle.dump(data_to_save, f_sim)

    print("\n🎉 V9 (修复 min_df=1) 大脑全部重建完毕！")


if __name__ == "__main__":
    build_and_save_matrix()