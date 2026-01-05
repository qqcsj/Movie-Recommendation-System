# Create your views here.
import random
import pickle
import jieba  # ⭐️ 1. 导入 jieba
from sklearn.metrics.pairwise import cosine_similarity
from django.core.cache import cache
from django.db.models import Count
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
# (新导入) 从 Django 设置中读取密钥
from django.conf import settings
from api.serializers import *
from cache_keys import USER_CACHE
# 下面一行就是“关键技术”！它从 recommend_movies.py 文件中，导入了两个真正的“推荐算法”函数
from recommend_movies import recommend_by_user_id, recommend_by_item_id
from user.models import Rate, Movie, Comment, User

# !! 新增导入 !!
from django.http import JsonResponse
from django.db.models import Q
# !! 新增导入结束 !!

from django.http import JsonResponse
from rest_framework.decorators import api_view

# (添加到 views.py 的 import 区域)

# ... 现有的导入 ...
# ⭐️⭐️⭐️ 必须添加下面这一行，否则 np.linspace 会报错 ⭐️⭐️⭐️
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

import json
import os


# ⭐️ 关键：设置 Matplotlib 为非交互模式，防止在服务器端报错
import matplotlib
matplotlib.use('Agg')
# 设置中文字体 (尝试使用常见中文字体，防止乱码)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


from zhipuai import ZhipuAI # 确保你已添加这行导入
# --- ⭐️ 配置你的 LLM 密钥 (智谱AI版) - 更安全的方式 ⭐️ ---
# (我们从 settings.py 中读取密钥)
try:
    # 尝试从 settings.py 获取密钥
    api_key = getattr(settings, 'ZHIPU_API_KEY', None)

    if api_key:
        client = ZhipuAI(api_key=api_key)
        llm_model_name = "glm-4-flash"  # 使用最新的 Flash 模型，速度快
        print("--- 🧠 LLM (智谱AI) 大脑已连接 ---")
    else:
        print("--- ⚠️ LLM (智谱AI) 密钥未在 settings.py 中配置 ---")
        client = None

except Exception as e:
    print(f"--- ⚠️ LLM (智谱AI) 连接失败: {e} ---")
    client = None


# (在 views.py 中)

# --- ⭐️ V4：“更智能”的 LLM 辅助函数 (智谱AI版) ⭐️ ---
def analyze_search_query(query_text):
    """
    使用 智谱AI LLM 分析用户查询, 提取关键字并判断“意图”。
    返回: (intent, clean_keywords)
    """
    if not client:
        print("LLM (智谱AI) 未初始化，退回到 'FIND_PLOT' 意图。")
        return 'FIND_PLOT', query_text  # 默认使用 TF-IDF 搜索

    # 提示词(Prompt) - V4 智能版
    prompt_instructions = """
    你是一个电影搜索引擎的助手。你的任务是分析用户的查询，判断用户的“意图”并提取“核心搜索关键字”。

    意图分为五类：
    1. `FIND_DIRECTOR`: 当用户在寻找一个“导演”的电影时。(例如: "推荐陈凯歌的电影", "张艺谋")
    2. `FIND_ACTOR`: 当用户在寻找一个“演员”的电影时。(例如: "周星驰的电影", "黄渤")
    3. `FIND_GENRE`: 当用户在寻找一个“类型”或“标签”的电影时。(例如: "推荐科幻片", "喜剧电影")
    4. `FIND_SPECIFIC`: 当用户在寻找一个“特定的”电影时。这通常发生在用户询问“某个角色”、“某个情节”或“某句台词”时。(例如: "千寻是哪个电影的角色？", "有句话叫'做人没梦想和咸鱼有什么分别'")
    5. `FIND_PLOT`: 当用户在寻找“相关”电影或描述一个“场景”时。这是“FIND_SPECIFIC”的补充，用于模糊搜索。(例如: "推荐关于机器人的电影", "一个男孩和一条龙")

    规则：
    1. 你的回答必须严格按照 "Intent: KEYWORD" 的格式。
    2. 不要说任何额外的话。
    3. `FIND_DIRECTOR`, `FIND_ACTOR`, `FIND_GENRE` 优先于 `FIND_PLOT`。
    4. 只有当用户在问“某个角色/情节”时，才使用 `FIND_SPECIFIC`。

    示例：
    用户查询: "推荐宫崎骏的电影"
    你的回答: FIND_DIRECTOR: 宫崎骏

    用户查询: "千寻是哪个电影的角色？"
    你的回答: FIND_SPECIFIC: 千寻

    用户查询: "陈凯歌"
    你的回答: FIND_DIRECTOR: 陈凯歌

    用户查询: "推荐科幻片"
    你的回答: FIND_GENRE: 科幻

    用户查询: "关于机器人的电影"
    你的回答: FIND_PLOT: 机器人

    用户查询: "周星驰演的"
    你的回答: FIND_ACTOR: 周星驰
    ---
    """

    try:
        response = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompt_instructions},
                {"role": "user", "content": f"用户查询: \"{query_text}\"\n你的回答:"}
            ],
            temperature=0.1
        )

        result = response.choices[0].message.content.strip()

        if ':' not in result:
            print(f"LLM 格式错误，退回: {result}")
            return 'FIND_PLOT', result

        intent, clean_keywords = result.split(':', 1)
        intent = intent.strip()
        clean_keywords = clean_keywords.strip()

        valid_intents = ['FIND_DIRECTOR', 'FIND_ACTOR', 'FIND_GENRE', 'FIND_SPECIFIC', 'FIND_PLOT']
        if intent not in valid_intents:
            print(f"LLM 意图错误，退回: {intent}")
            intent = 'FIND_PLOT'

        print(f"LLM V4 (智谱) 分析：'{query_text}' -> 意图: '{intent}', 关键字: '{clean_keywords}'")
        return intent, clean_keywords
    except Exception as e:
        print(f"LLM (智谱) 提取关键字失败: {e}")
        return 'FIND_PLOT', query_text


# --- ⭐️ V4：“更智能”的 AI 助手接口 ⭐️ ---
@csrf_exempt
@api_view(['POST'])
def ai_search(request):
    """
    V4：由 LLM 驱动的智能 AI 助手接口 (区分精确查询和模糊查询)
    """
    query_text = request.data.get('query', '')
    if not query_text:
        return Response([])

    # 【第 1 步：调用“智能” LLM V4 分析意图和关键字】
    intent, clean_query = analyze_search_query(query_text)

    final_movies = []

    try:
        # ⭐️【V4 核心逻辑：按意图分流】⭐️

        if intent == 'FIND_DIRECTOR':
            # 1. 精确查询：导演 (100% 准确, 按评分排序)
            """1.director:这是 Movie 模型中定义的一个字段名（数据库表的一列）
                2.icontains:这是查询动作，全称是 Case-Insensitive Contains（不区分大小写的包含）。
                i: Insensitive，表示忽略大小写（例如输入 "cameron" 也能搜到 "James Cameron"）。
                contains: 包含，表示模糊匹配（只要字段里包含这段文字就行，不需要完全相等）。
                =clean_query:把用户输入的关键字（clean_query）作为查询的目标值。"""
            movies_queryset = Movie.objects.filter(director__icontains=clean_query)
            # 2. 排序与切片：按 d_rate (评分) 从高到低排序，并只取前 10 个
            final_movies = movies_queryset.order_by('-d_rate')[:10]
            print(f"--- 诊断: 意图 '{intent}', 数据库精确查询 (Director) 找到 {len(final_movies)} 部。")

        elif intent == 'FIND_ACTOR':
            # 2. 精确查询：演员 (100% 准确, 按评分排序)
            movies_queryset = Movie.objects.filter(leader__icontains=clean_query)  # 假设 'leader' 是演员字段
            final_movies = movies_queryset.order_by('-d_rate')[:10]
            print(f"--- 诊断: 意图 '{intent}', 数据库精确查询 (Actor) 找到 {len(final_movies)} 部。")

        elif intent == 'FIND_GENRE':
            # 3. 精确查询：类型 (100% 准确, 按评分排序)
            movies_queryset = Movie.objects.filter(tags__name__icontains=clean_query).distinct()
            final_movies = movies_queryset.order_by('-d_rate')[:10]
            print(f"--- 诊断: 意图 '{intent}', 数据库精确查询 (Genre) 找到 {len(final_movies)} 部。")

        else:
            # 4. 模糊查询 (TF-IDF)：用于 'FIND_SPECIFIC' (千寻) 和 'FIND_PLOT' (机器人)

            print(f"--- 诊断: 意图 '{intent}', 转入 TF-IDF 模糊搜索...")

            # ⭐️ 2. 【V9 中文分词修复】 ⭐️
            #    (我们必须同样对用户的查询进行分词)
            #加空格？ 因为英文是用空格分词的，为了适配后面的算法（本来是给英文设计的），我们需要把中文伪装成空格隔开的样子。
            tokenized_query = ' '.join(jieba.cut_for_search(clean_query))
            print(f"--- 诊断: 关键字 '{clean_query}' 分词为 -> '{tokenized_query}'")

            # 3. 将“分词后”的关键字转换为“TF-IDF 向量”
            #    (注意：这里用的是 tokenized_query, 而不是 clean_query)
            query_vector = TFIDF_VECTORIZER.transform([tokenized_query])

            # 4. 计算“余弦相似度”
            sim_scores = cosine_similarity(query_vector, TFIDF_MATRIX)
            sim_scores_list = sim_scores[0]

            # 5. 找到所有 > 0.1 的匹配项的“索引” (V8 -> V9: 此处逻辑不变)
            top_indices = sim_scores_list.argsort()[-20:][::-1]
            """
            1.[-20:]：切片，只要最后 20 个。因为是从小到大排的，最后 20 个就是分数最高的 20 个。
            2.[::-1]：倒序。把这 20 个反过来，变成从大到小。也就是第一名排在最前面。
            """
            top_scores = [sim_scores_list[i] for i in top_indices[:5]]
            print(f"--- 诊断: 关键字 '{clean_query}' (分词后) 的 Top 5 相似度得分: {top_scores}")

            relevant_indices = [i for i in top_indices if sim_scores_list[i] > 0.1]

            """
            逻辑：即便排在第一名，如果相似度只有 0.001，那也是不相关的。
            > 0.1：这是一个门槛（Threshold）。只有相似度大于 0.1 的电影才会被留下来。
            如果所有的电影得分都低于 0.1，说明用户搜的东西库里压根没有。
            """

            if not relevant_indices:
                print(f"关键字 '{clean_query}' 未找到 > 0.1 的匹配项。")
                return Response([])
            #如果过滤完一个都不剩，直接告诉前端“没找到”，返回空列表，避免报错。

            print(f"--- 诊断: TF-IDF 找到的 relevant_indices (索引): {relevant_indices}")

            if not ID_MOVIE_MAP:
                print("--- 致命错误: ID_MOVIE_MAP (来自 content_sim.pkl) 未加载或为空。")
                return Response([])
            #检查一下“地图”是不是没加载。如果地图丢了，只有行号也没用

            movie_ids = [ID_MOVIE_MAP[i] for i in relevant_indices if i in ID_MOVIE_MAP]
            print(f"--- 诊断: 转换后的 movie_ids: {movie_ids}")

            if not movie_ids:
                print(f"--- 诊断: 索引 {relevant_indices} 在 ID_MOVIE_MAP 中一个也未找到。")
                return Response([])

            # ⭐️【修复排序 Bug】⭐️
            # 我们必须先按“相似度”排序 (因为 TF-IDF 是按相关性找的)
            score_map = {ID_MOVIE_MAP[i]: sim_scores_list[i] for i in relevant_indices if i in ID_MOVIE_MAP}

            # 先从数据库一次性取出
            movies_queryset = Movie.objects.filter(id__in=movie_ids)

            # 用 Python 排序 (sorted), 而不是 .order_by(),
            # 因为我们要按“相似度得分” (score_map) 来排
            movies_sorted_by_sim = sorted(list(movies_queryset), key=lambda m: score_map.get(m.id, 0), reverse=True)

            if intent == 'FIND_SPECIFIC':
                # (例如 "千寻") -> 只返回最相关的 1 部
                if movies_sorted_by_sim:
                    final_movies = movies_sorted_by_sim[0:1]
            else:  # intent == 'FIND_PLOT'
                # (例如 "机器人") -> 返回相关列表 (最多 10 部)
                final_movies = movies_sorted_by_sim[:10]

            # 7. 序列化并返回
        serializer = MovieSerializer(final_movies, many=True)
        return Response(serializer.data)

    except Exception as e:
        print(f"AI 搜索 V9 失败：: {e}")
        import traceback
        traceback.print_exc()
        return Response([])


# AI大脑，“AI 大脑加载区” 必须放在“这里”！
print("--- 正在加载“AI 翻译官”(tfidf_vec.pkl)... ---")
try:
    TFIDF_VECTORIZER = pickle.load(open('tfidf_vec.pkl', 'rb'))

    print("--- 正在加载“电影特征矩阵”(tfidf_matrix.pkl)... ---")
    TFIDF_MATRIX = pickle.load(open('tfidf_matrix.pkl', 'rb'))

    # ⭐️【V7 最终修复 - 匹配3文件版】⭐️

    # 1. content_sim.pkl 包含一个 (矩阵, 地图) 的元组
    print("--- 正在加载“内容相似度”大脑 (content_sim.pkl)... ---")
    CONTENT_SIM_DATA = pickle.load(open('content_sim.pkl', 'rb'))

    # 2. 解包这个元组
    CONTENT_SIM_MATRIX = CONTENT_SIM_DATA[0]  # 第0项是矩阵
    MOVIE_ID_MAP = CONTENT_SIM_DATA[1]  # 第1项是 {movie_id: index} 字典

    # 3. 我们需要“反向地图” {index: movie_id}
    # (这行代码现在 100% 正确了)
    ID_MOVIE_MAP = {index: movie_id for movie_id, index in MOVIE_ID_MAP.items()}

    print("--- 🎉 AI 大脑 (V7) 加载完毕！ ---")
except Exception as e:
    print(f" !!! AI 大G.O. 大脑加载失败: {e} !!!")
    import traceback

    traceback.print_exc()
    TFIDF_VECTORIZER = None
    TFIDF_MATRIX = None
    CONTENT_SIM_MATRIX = None
    MOVIE_ID_MAP = None
    ID_MOVIE_MAP = None

#“我的评分”API，供 'personal.html' 页面（个人中心）的 "My ratings" 按钮调用
"""
@api_view(['GET'])：装饰器。告诉 DRF 这个 rate_detail 函数是一个 API 接口，并且只允许 HTTP GET 请求访问它
"""
@api_view(['GET'])
def rate_detail(request, user_id):
    """
    request 是一个 Python 对象，它包含了前端这次请求的所有信息，
    用的是什么方法？（request.method，比如 GET；他是谁？（request.user，如果是登录用户；他还带了什么数据？（request.data
    """
    rate = Rate.objects.filter(user_id=user_id)
    # a=Rate.objects.first()
    serializer = RateSerializer(rate, many=True)#传给它的是一个列表（多条记录），而不是单个对象
    return Response(serializer.data)

# “我的收藏”API，供 'personal.html' 页面的 "My favourites" 按钮调用
@api_view(['GET'])
def collect_detail(request, user_id):
    user = User.objects.get(id=user_id)
    collect_movies = user.movie_set.all()
    serializer = CollectSerializer(collect_movies, many=True)
    return Response(serializer.data)

#“我的评论”API，供 'personal.html' 页面的 "My comments" 按钮调用
@api_view(['GET'])
def comment_detail(request, user_id):
    rate = Comment.objects.filter(user_id=user_id)
    serializer = CommentSerializer(rate, many=True)
    return Response(serializer.data)


"""
1.@api_view(['GET']) 放在一个函数定义之前(比如 my_view)，就等同于在函数定义之后写 my_view = api_view(['GET'])(my_view)。
它的作用是包装下面的函数，为它添加额外的功能
2.api_view (DRF 的功能),是 DRF 提供的核心装饰器之一，用于基于函数的视图 (Function-Based Views)。
它将一个普通的 Django 视图函数转换成一个 DRF 的 APIView。
3.(['GET']) (参数：允许的方法),传递给 api_view 装饰器的参数，它是一个列表，规定了该视图允许处理的 HTTP 方法。['GET'] 表示：只允许 GET 请求。
"""
@api_view(['GET'])
def user_recommend(request, user_id=None):
    """
    基于用户的协同过滤 API 接口。
    当首页的 Vue.js (前端）脚本访问 /api/user_recommend/[用户ID] 时，这个函数就会被触发，会接收 DRF 传来的 'request' 对象，
    会从 URL 中接收一个 'user_id'
    """
    if user_id is None:
        # 是游客，不是登录用户，从 Movie (电影) 数据库表中，随机('.order_by('?')')取一些电影。
        movie_list = Movie.objects.order_by('?')
    else:

        # 是登录用户，尝试从"缓存"中获取数据 (为了速度)
        # 引入“缓存”技术 (优化性能)，算法很慢，所以我们用 cache (缓存) 来“抄近路”。先检查缓存里有没有这个 user_id 的“旧”推荐结果_
        cache_key = USER_CACHE.format(user_id=user_id)
        movie_list = cache.get(cache_key)
        if movie_list is None:
            #缓存“未命中”，缓存里没有，说明是“新”用户，我们只能“现场”去跑那个很慢的推荐算法
            movie_list = recommend_by_user_id(user_id)
            #写入“缓存”，把“现场”算出来的结果，存进缓存里 5 分钟 (60*5)，这样他下次再来，就不用再算了_
            cache.set(cache_key, movie_list, 60 * 5)
            print('设置缓存')
        else:
            print('缓存命中!')#如果缓存里有，就直接“抄近路”，跳过算法
    movie_list = list(movie_list)
    random.shuffle(movie_list)
    #增加随机性，为了让用户每次刷新都能看到点“新东西”，把推荐结果随机打乱一下_
    serializer = MovieSerializer(movie_list, many=True)#把 Python 对象转成 JSON 文本
    return Response(serializer.data)#（通过 API 把 JSON 文本发回给前端 Vue.js


@api_view(['GET'])
def item_recommend(request, user_id=None):
    if user_id is None:
        movie_list = Movie.objects.order_by('?')
    else:
        cache_key = USER_CACHE.format(user_id=user_id)
        movie_list = cache.get(cache_key)
        if movie_list is None:
            movie_list = recommend_by_item_id(user_id)
            cache.set(cache_key, movie_list, 60 * 5)
            print('设置缓存')
        else:
            print('缓存命中!')
    movie_list = list(movie_list)
    random.shuffle(movie_list)
    serializer = MovieSerializer(movie_list, many=True)
    return Response(serializer.data)


#“热门电影” API，这是一个“更简单”的推荐算法：不是“猜你喜欢”，而是“推荐大家都在看的热门”
@api_view(['GET'])
def hotest_movie(request):
    #.annotate(...) 负责“计数”（统计每部电影被多少人收藏）
    #.order_by('-user_collector') 负责“倒序”排列（被收藏越多的排越前）
    #[:10] 负责“切片”（只选出前 10 名）
    movies = Movie.objects.annotate(user_collector=Count('collect')).order_by('-user_collector')[:10]
    serializer = MovieSerializer(movies, many=True)
    return Response(serializer.data)


# ⭐️【内容（相关）推荐】(已修复排序 Bug 和清理) ⭐️
@api_view(['GET'])
def content_recommend(request, movie_id):
    """
    基于内容的推荐 API 接口 (相关推荐)
    """
    if CONTENT_SIM_MATRIX is None or not MOVIE_ID_MAP:
        print("内容推荐失败：AI 大脑未加载")
        return Response([])

    try:
        # 1. 从“全局 ID 地图”里找到这部电影在“矩阵”里的“行号”
        movie_index = MOVIE_ID_MAP[movie_id]

        # 2. 从“矩阵”里拿到这一行所有的“相似度得分”
        sim_scores = list(enumerate(CONTENT_SIM_MATRIX[movie_index]))

        # 3. 按照“得分”倒序排列
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # 4. 选出 Top 5 (第 0 个是它自己，所以我们选 1 到 6)
        top_5_indices = [i[0] for i in sim_scores[1:6]]

        # 5. 从“全局反向地图”里反向查出这些“行号”对应的“电影ID”
        #    (我们使用全局的 ID_MOVIE_MAP)
        top_5_movie_ids = [ID_MOVIE_MAP[i] for i in top_5_indices if i in ID_MOVIE_MAP]

        # 6.【⭐️ 修复排序 Bug ⭐️】
        #    使用 .in_bulk() 高效查询，并保持“相关性”顺序
        movies_dict = Movie.objects.in_bulk(top_5_movie_ids)
        recommend_list = [movies_dict[movie_id] for movie_id in top_5_movie_ids if movie_id in movies_dict]

    except KeyError:
        # 捕获当 movie_id 不在地图中的错误
        print(f"内容推荐警告：电影 {movie_id} 不在 AI 地图中。")
        recommend_list = Movie.objects.order_by('?')[:5] # 随机返回
    except Exception as e:
        print(f"内容推荐算法出错: {e}")
        # 如果出错了，就随便返回 5 部电影，防止网页崩溃
        recommend_list = Movie.objects.order_by('?')[:5]

    serializer = MovieSerializer(recommend_list, many=True)
    return Response(serializer.data)


# !! ========================================== !!
# !!               新增的抽奖 API                !!
# !! ========================================== !!

# 这是一个普通的 Django 视图，不是 DRF API，所以不需要 @api_view
# 我们使用 JsonResponse，它更轻量
def draw_movie_api(request):
    category = request.GET.get('category')

    # 1. 定义评分范围
    #    (根据您的 serializers.py, 评分字段是 d_rate)
    """
    1.d_rate：这是你的数据库模型（Movie 表）中的字段名。在这个项目中，它代表电影评分。
    2.__ (双下划线)：这是 Django 框架的特殊语法，用来连接“字段名”和“查询条件”。
    3.gte：这是 Greater Than or Equal to 的缩写，意思是 “大于或等于” (>=)。
      lt：是 Less Than 的缩写，意思是 “小于” (<)。
   4.Q 是 Django 框架自带的一个类。用它是为了把查询条件封装成对象，存在字典里。
    可以根据前端传来的不同标签（如‘优秀’或‘烂片’），灵活地从字典里取出对应的查询条件
    """
    rating_filters = {
        'excellent': Q(d_rate__gte=8.5),
        'recommended': Q(d_rate__gte=7.5, d_rate__lt=8.5),
        'pass': Q(d_rate__gte=6.5, d_rate__lt=7.5),
        'mediocre': Q(d_rate__gte=5.5, d_rate__lt=6.5),
        'bad': Q(d_rate__lt=5.5),
    }
    #从 rating_filters 这个（字典）里，根据 category 这个（键），
    # 把对应的“查询条件”（Q对象/值）取出来，赋值给变量 filter_q。
    filter_q = rating_filters.get(category)

    if not filter_q:
        return JsonResponse({'error': 'Invalid category'}, status=400)
    """
    JsonResponse(..., status=400) —— 发送错误报告
    JsonResponse({'error': 'Invalid category'})：服务器给前端（网页）回信，内容是一段 JSON 数据：{"error": "Invalid category"}。
    
    前端收到后，JS 代码就能看到这个错误提示。
    """

    # 2. 从数据库查询所有符合条件的电影 ID
    #    .values_list('id', flat=True) 是最高效的查询方式
    #    (确保 Movie 模型已从 user.models 导入)
    """
    1.Movie.objects.filter(filter_q)：筛选。先根据刚才的条件（比如“烂片”），把不符合的电影过滤掉，只留下符合的。
    2..values_list('id', ...)：只取特定列。告诉数据库：“我只要 id 这一列，别的字段（name, image_link, d_rate）我都不要。”
    如果不加 flat=True，拿到的结果是像这样的元组列表：[(1,), (2,), (5,)]。这就像是每个人虽然只报了学号，但每个人都还单独包了一个信封，拆起来麻烦。
    flat=True：扁平化。这是关键！它的意思是：“把那些信封都拆了，直接给我数字。”结果就变成了干净的列表：[1, 2, 5]。
    3.list(...)：转成列表。
    Django 查询出来的结果默认是一个“查询集”，
    为了方便后面用 random.choice() 进行随机抽取，我们把它强制转换成 Python 最普通的列表。
    """
    movie_ids = list(Movie.objects.filter(filter_q).values_list('id', flat=True))

    if not movie_ids:
        # 如果这个分类一部电影都没有
        return JsonResponse({'error': 'No movies found for this category'}, status=404)

    # 3. 随机选择一个 ID
    random_id = random.choice(movie_ids)

    # 4. 获取完整的电影对象
    try:
        # 尝试：拿着刚才随机选中的号码 (random_id)，去数据库里找那部电影的完整对象
        movie = Movie.objects.get(id=random_id)
    except Movie.DoesNotExist:
        # 异常处理：万一（虽然概率很低）这个号码对应的电影刚刚被删了，找不到了
        # 就返回一个 404 错误，告诉前端“没找到”
        return JsonResponse({'error': 'Movie not found'}, status=404)

    # 手动序列化
    # 5. 序列化为 JSON
    #    (根据您的 serializers.py, 字段名为 name, d_rate, image_link)
    #    (前端 JS 需要 'id', 'name', 'rating', 'image_link')
    data = {
        'id': movie.id,
        'name': movie.name,
        'rating': movie.d_rate,  # JS 需要 'rating', 对应您的 d_rate 字段(前端 JS 代码里（_movie_lottery.html）想要接收的字段叫 rating。)
        'image_link': movie.image_link.url if movie.image_link else None  # JS 需要 'image_link',如果这部电影有海报 (if movie.image_link)，就取它的 URL 链接 (.url)；如果没有海报（比如忘了上传），就给一个 None（空值）。
    }

    #data：刚才打包好的那个 Python 字典
    #JsonResponse：Django 的快递员。它会把 Python 字典转换成 JSON 字符串（一种所有网页都认识的文本格式），然后发送给浏览器
    return JsonResponse(data)


# --- ⭐️ 投票系统数据 (全局变量) ⭐️ ---
# 为了方便，我们把20个人物直接定义在这里
CHARACTER_DATA = [
    # ==========================================
    # 1. 2025年 1-2月 (春节档/寒假档 预测与定档)
    # ==========================================
    {"id": 1, "name": "郭靖 (肖战饰)", "movie": "射雕英雄传：侠之大者 (2025)", "likes": 0, "dislikes": 0},
    {"id": 2, "name": "黄蓉 (庄达菲饰)", "movie": "射雕英雄传：侠之大者 (2025)", "likes": 0, "dislikes": 0},
    {"id": 3, "name": "哪吒", "movie": "哪吒之魔童闹海 (2025)", "likes": 0, "dislikes": 0},
    {"id": 4, "name": "敖丙", "movie": "哪吒之魔童闹海 (2025)", "likes": 0, "dislikes": 0},
    {"id": 5, "name": "姬发", "movie": "封神第二部：战火西岐 (2025)", "likes": 0, "dislikes": 0},
    {"id": 6, "name": "殷寿", "movie": "封神第二部：战火西岐 (2025)", "likes": 0, "dislikes": 0},
    {"id": 7, "name": "姜子牙", "movie": "封神第二部：战火西岐 (2025)", "likes": 0, "dislikes": 0},
    {"id": 8, "name": "邓婵玉", "movie": "封神第二部：战火西岐 (2025)", "likes": 0, "dislikes": 0},
    {"id": 9, "name": "唐仁", "movie": "唐探1900 (2025)", "likes": 0, "dislikes": 0},
    {"id": 10, "name": "秦风", "movie": "唐探1900 (2025)", "likes": 0, "dislikes": 0},
    {"id": 11, "name": "崔文子", "movie": "蛟龙行动 (2025)", "likes": 0, "dislikes": 0},
    {"id": 12, "name": "高豪", "movie": "蛟龙行动 (2025)", "likes": 0, "dislikes": 0},
    {"id": 13, "name": "帕丁顿熊", "movie": "柏灵顿：熊熊去秘鲁 (2025)", "likes": 0, "dislikes": 0},
    {"id": 14, "name": "布朗先生", "movie": "柏灵顿：熊熊去秘鲁 (2025)", "likes": 0, "dislikes": 0},
    {"id": 15, "name": "米奇", "movie": "米奇17 (2025)", "likes": 0, "dislikes": 0},
    {"id": 16, "name": "美国队长 (山姆)", "movie": "美国队长4 (2025)", "likes": 0, "dislikes": 0},

    # ==========================================
    # 2. 熊出没宇宙 (大电影全家桶)
    # ==========================================
    {"id": 17, "name": "熊大", "movie": "熊出没·重启未来 (2025)", "likes": 0, "dislikes": 0},
    {"id": 18, "name": "熊二", "movie": "熊出没·重启未来 (2025)", "likes": 0, "dislikes": 0},
    {"id": 19, "name": "光头强", "movie": "熊出没·重启未来 (2025)", "likes": 0, "dislikes": 0},
    {"id": 20, "name": "吉吉国王", "movie": "熊出没·重返地球", "likes": 0, "dislikes": 0},
    {"id": 21, "name": "毛毛", "movie": "熊出没·重返地球", "likes": 0, "dislikes": 0},
    {"id": 22, "name": "蹦蹦", "movie": "熊出没之雪岭熊风", "likes": 0, "dislikes": 0},
    {"id": 23, "name": "涂涂 (猫头鹰)", "movie": "熊出没之雪岭熊风", "likes": 0, "dislikes": 0},
    {"id": 24, "name": "团子 (山神)", "movie": "熊出没之雪岭熊风", "likes": 0, "dislikes": 0},
    {"id": 25, "name": "纳雅", "movie": "熊出没·奇幻空间", "likes": 0, "dislikes": 0},
    {"id": 26, "name": "乐天", "movie": "熊出没·狂野大陆", "likes": 0, "dislikes": 0},
    {"id": 27, "name": "小可", "movie": "熊出没·狂野大陆", "likes": 0, "dislikes": 0},
    {"id": 28, "name": "阿布", "movie": "熊出没·重返地球", "likes": 0, "dislikes": 0},
    {"id": 29, "name": "赵琳", "movie": "熊出没·探险日记", "likes": 0, "dislikes": 0},
    {"id": 30, "name": "天才威", "movie": "熊出没·怪兽计划", "likes": 0, "dislikes": 0},
    {"id": 31, "name": "李老板", "movie": "熊出没 (系列)", "likes": 0, "dislikes": 0},
    {"id": 32, "name": "肥波", "movie": "熊出没 (系列)", "likes": 0, "dislikes": 0},

    # ==========================================
    # 3. 喜羊羊与灰太狼宇宙
    # ==========================================
    {"id": 33, "name": "喜羊羊", "movie": "喜羊羊之守护 (2024)", "likes": 0, "dislikes": 0},
    {"id": 34, "name": "灰太狼", "movie": "喜羊羊之守护 (2024)", "likes": 0, "dislikes": 0},
    {"id": 35, "name": "美羊羊", "movie": "喜羊羊之筐出未来", "likes": 0, "dislikes": 0},
    {"id": 36, "name": "懒羊羊", "movie": "喜羊羊之筐出未来", "likes": 0, "dislikes": 0},
    {"id": 37, "name": "沸羊羊", "movie": "喜羊羊之筐出未来", "likes": 0, "dislikes": 0},
    {"id": 38, "name": "暖羊羊", "movie": "喜羊羊之筐出未来", "likes": 0, "dislikes": 0},
    {"id": 39, "name": "慢羊羊", "movie": "喜羊羊之筐出未来", "likes": 0, "dislikes": 0},
    {"id": 40, "name": "红太狼", "movie": "喜羊羊之筐出未来", "likes": 0, "dislikes": 0},
    {"id": 41, "name": "小灰灰", "movie": "喜羊羊之筐出未来", "likes": 0, "dislikes": 0},
    {"id": 42, "name": "蕉太狼", "movie": "喜羊羊与灰太狼", "likes": 0, "dislikes": 0},
    {"id": 43, "name": "机械羊", "movie": "喜羊羊之羊羊运动会", "likes": 0, "dislikes": 0},
    {"id": 44, "name": "潇洒哥", "movie": "喜羊羊之古古怪界", "likes": 0, "dislikes": 0},
    {"id": 45, "name": "黑大帅", "movie": "喜羊羊之古古怪界", "likes": 0, "dislikes": 0},
    {"id": 46, "name": "虎威太岁", "movie": "喜羊羊之虎虎生威", "likes": 0, "dislikes": 0},
    {"id": 47, "name": "小乖乖", "movie": "喜羊羊之牛气冲天", "likes": 0, "dislikes": 0},
    {"id": 48, "name": "智羊羊", "movie": "喜羊羊之跨时空救兵", "likes": 0, "dislikes": 0},

    # ==========================================
    # 4. 2024年 7-8月 (暑期档热片)
    # ==========================================
    {"id": 49, "name": "马继业", "movie": "抓娃娃", "likes": 0, "dislikes": 0},
    {"id": 50, "name": "马成钢", "movie": "抓娃娃", "likes": 0, "dislikes": 0},
    {"id": 51, "name": "春兰", "movie": "抓娃娃", "likes": 0, "dislikes": 0},
    {"id": 52, "name": "死侍 (韦德)", "movie": "死侍与金刚", "likes": 0, "dislikes": 0},
    {"id": 53, "name": "金刚狼 (罗根)", "movie": "死侍与金刚", "likes": 0, "dislikes": 0},
    {"id": 54, "name": "小白", "movie": "白蛇：浮生", "likes": 0, "dislikes": 0},
    {"id": 55, "name": "许仙", "movie": "白蛇：浮生", "likes": 0, "dislikes": 0},
    {"id": 56, "name": "小青", "movie": "白蛇：浮生", "likes": 0, "dislikes": 0},
    {"id": 57, "name": "雷恩 (Rain)", "movie": "异形：夺命舰", "likes": 0, "dislikes": 0},
    {"id": 58, "name": "安迪 (Andy)", "movie": "异形：夺命舰", "likes": 0, "dislikes": 0},
    {"id": 59, "name": "高志垒", "movie": "逆行人生", "likes": 0, "dislikes": 0},
    {"id": 60, "name": "容金珍", "movie": "解密", "likes": 0, "dislikes": 0},
    {"id": 61, "name": "张楚岚", "movie": "异人之下", "likes": 0, "dislikes": 0},
    {"id": 62, "name": "冯宝宝", "movie": "异人之下", "likes": 0, "dislikes": 0},
    {"id": 63, "name": "格鲁", "movie": "神偷奶爸4", "likes": 0, "dislikes": 0},
    {"id": 64, "name": "小黄人", "movie": "神偷奶爸4", "likes": 0, "dislikes": 0},

    # ==========================================
    # 5. 2024年 1月 (元旦/寒假档回顾)
    # ==========================================
    {"id": 65, "name": "杜乐莹", "movie": "热辣滚烫", "likes": 0, "dislikes": 0},
    {"id": 66, "name": "昊坤", "movie": "热辣滚烫", "likes": 0, "dislikes": 0},
    {"id": 67, "name": "张驰", "movie": "飞驰人生2", "likes": 0, "dislikes": 0},
    {"id": 68, "name": "厉小海", "movie": "飞驰人生2", "likes": 0, "dislikes": 0},
    {"id": 69, "name": "孙宇强", "movie": "飞驰人生2", "likes": 0, "dislikes": 0},
    {"id": 70, "name": "韩明", "movie": "第二十条", "likes": 0, "dislikes": 0},
    {"id": 71, "name": "李茂娟", "movie": "第二十条", "likes": 0, "dislikes": 0},
    {"id": 72, "name": "程程", "movie": "金手指", "likes": 0, "dislikes": 0},
    {"id": 73, "name": "庄辉", "movie": "年会不能停！", "likes": 0, "dislikes": 0},
    {"id": 74, "name": "胡建林", "movie": "年会不能停！", "likes": 0, "dislikes": 0},
    {"id": 75, "name": "林阵安", "movie": "潜行", "likes": 0, "dislikes": 0},
    {"id": 76, "name": "修浩", "movie": "潜行", "likes": 0, "dislikes": 0},

    # ==========================================
    # 6. 2025年 7-8月 (暑期档 展望)
    # ==========================================
    {"id": 77, "name": "超人 (克拉克)", "movie": "超人 (2025)", "likes": 0, "dislikes": 0},
    {"id": 78, "name": "路易丝", "movie": "超人 (2025)", "likes": 0, "dislikes": 0},
    {"id": 79, "name": "神奇先生", "movie": "神奇四侠 (2025)", "likes": 0, "dislikes": 0},
    {"id": 80, "name": "隐形女侠", "movie": "神奇四侠 (2025)", "likes": 0, "dislikes": 0},
    {"id": 81, "name": "霹雳火", "movie": "神奇四侠 (2025)", "likes": 0, "dislikes": 0},
    {"id": 82, "name": "石头人", "movie": "神奇四侠 (2025)", "likes": 0, "dislikes": 0},
    {"id": 83, "name": "大坏狼", "movie": "坏蛋联盟2", "likes": 0, "dislikes": 0},
    {"id": 84, "name": "灶门炭治郎", "movie": "鬼灭之刃: 无限城篇", "likes": 0, "dislikes": 0},
    {"id": 85, "name": "鬼舞辻无惨", "movie": "鬼灭之刃: 无限城篇", "likes": 0, "dislikes": 0},
    {"id": 86, "name": "李善德", "movie": "长安的荔枝", "likes": 0, "dislikes": 0},
    {"id": 87, "name": "佐拉", "movie": "侏罗纪世界: 重生", "likes": 0, "dislikes": 0},
    {"id": 88, "name": "杨贵妃", "movie": "长安的荔枝", "likes": 0, "dislikes": 0},

    # ==========================================
    # 7. 宫崎骏吉卜力 (补全)
    # ==========================================
    {"id": 89, "name": "千寻", "movie": "千与千寻", "likes": 0, "dislikes": 0},
    {"id": 90, "name": "哈尔", "movie": "哈尔的移动城堡", "likes": 0, "dislikes": 0},
    {"id": 91, "name": "龙猫", "movie": "龙猫", "likes": 0, "dislikes": 0},
    {"id": 92, "name": "波妞", "movie": "悬崖上的金希儿", "likes": 0, "dislikes": 0},
    {"id": 93, "name": "无脸男", "movie": "千与千寻", "likes": 0, "dislikes": 0},
    {"id": 94, "name": "琪琪", "movie": "魔女宅急便", "likes": 0, "dislikes": 0},
    {"id": 95, "name": "吉吉 (黑猫)", "movie": "魔女宅急便", "likes": 0, "dislikes": 0},
    {"id": 96, "name": "珊 (幽灵公主)", "movie": "幽灵公主", "likes": 0, "dislikes": 0},
    {"id": 97, "name": "阿席达卡", "movie": "幽灵公主", "likes": 0, "dislikes": 0},
    {"id": 98, "name": "巴鲁", "movie": "天空之城", "likes": 0, "dislikes": 0},
    {"id": 99, "name": "希达", "movie": "天空之城", "likes": 0, "dislikes": 0},
    {"id": 100, "name": "宗介", "movie": "悬崖上的金希儿", "likes": 0, "dislikes": 0},
    {"id": 101, "name": "卡西法", "movie": "哈尔的移动城堡", "likes": 0, "dislikes": 0},
    {"id": 102, "name": "汤婆婆", "movie": "千与千寻", "likes": 0, "dislikes": 0},
    {"id": 103, "name": "白龙", "movie": "千与千寻", "likes": 0, "dislikes": 0},
    {"id": 104, "name": "小梅", "movie": "龙猫", "likes": 0, "dislikes": 0},
    {"id": 105, "name": "小月", "movie": "龙猫", "likes": 0, "dislikes": 0},
    {"id": 106, "name": "苍鹭", "movie": "你想活出怎样的人生", "likes": 0, "dislikes": 0},
    {"id": 107, "name": "牧真人", "movie": "你想活出怎样的人生", "likes": 0, "dislikes": 0},
    {"id": 108, "name": "红猪", "movie": "红猪", "likes": 0, "dislikes": 0},
    {"id": 109, "name": "娜乌西卡", "movie": "风之谷", "likes": 0, "dislikes": 0},

    # ==========================================
    # 8. 哈利波特魔法世界 (补全)
    # ==========================================
    {"id": 110, "name": "哈利·波特", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 111, "name": "赫敏·格兰杰", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 112, "name": "罗恩·韦斯莱", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 113, "name": "邓布利多", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 114, "name": "斯内普", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 115, "name": "伏地魔", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 116, "name": "德拉科·马尔福", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 117, "name": "海格", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 118, "name": "小天狼星", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 119, "name": "多比", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 120, "name": "纽特", "movie": "神奇动物在哪里", "likes": 0, "dislikes": 0},
    {"id": 121, "name": "嗅嗅", "movie": "神奇动物在哪里", "likes": 0, "dislikes": 0},

    # ==========================================
    # 9. 漫威/DC 超级英雄 (补全)
    # ==========================================
    {"id": 122, "name": "钢铁侠", "movie": "复仇者联盟", "likes": 0, "dislikes": 0},
    {"id": 123, "name": "蜘蛛侠", "movie": "蜘蛛侠", "likes": 0, "dislikes": 0},
    {"id": 124, "name": "美国队长", "movie": "复仇者联盟", "likes": 0, "dislikes": 0},
    {"id": 125, "name": "雷神", "movie": "复仇者联盟", "likes": 0, "dislikes": 0},
    {"id": 126, "name": "洛基", "movie": "复仇者联盟", "likes": 0, "dislikes": 0},
    {"id": 127, "name": "黑寡妇", "movie": "复仇者联盟", "likes": 0, "dislikes": 0},
    {"id": 128, "name": "绿巨人", "movie": "复仇者联盟", "likes": 0, "dislikes": 0},
    {"id": 129, "name": "鹰眼", "movie": "复仇者联盟", "likes": 0, "dislikes": 0},
    {"id": 130, "name": "奇异博士", "movie": "奇异博士", "likes": 0, "dislikes": 0},
    {"id": 131, "name": "灭霸", "movie": "复仇者联盟", "likes": 0, "dislikes": 0},
    {"id": 132, "name": "毒液", "movie": "毒液", "likes": 0, "dislikes": 0},
    {"id": 133, "name": "蝙蝠侠", "movie": "蝙蝠侠", "likes": 0, "dislikes": 0},
    {"id": 134, "name": "小丑 (Joker)", "movie": "蝙蝠侠", "likes": 0, "dislikes": 0},
    {"id": 135, "name": "神奇女侠", "movie": "正义联盟", "likes": 0, "dislikes": 0},
    {"id": 136, "name": "海王", "movie": "海王", "likes": 0, "dislikes": 0},
    {"id": 137, "name": "闪电侠", "movie": "闪电侠", "likes": 0, "dislikes": 0},
    {"id": 138, "name": "哈莉·奎茵", "movie": "自杀小队", "likes": 0, "dislikes": 0},

    # ==========================================
    # 10. 日漫/国漫 热门 (补全)
    # ==========================================
    {"id": 139, "name": "路飞", "movie": "海贼王", "likes": 0, "dislikes": 0},
    {"id": 140, "name": "索隆", "movie": "海贼王", "likes": 0, "dislikes": 0},
    {"id": 141, "name": "乔巴", "movie": "海贼王", "likes": 0, "dislikes": 0},
    {"id": 142, "name": "娜美", "movie": "海贼王", "likes": 0, "dislikes": 0},
    {"id": 143, "name": "山治", "movie": "海贼王", "likes": 0, "dislikes": 0},
    {"id": 144, "name": "艾斯", "movie": "海贼王", "likes": 0, "dislikes": 0},
    {"id": 145, "name": "红发香克斯", "movie": "海贼王", "likes": 0, "dislikes": 0},
    {"id": 146, "name": "漩涡鸣人", "movie": "火影忍者", "likes": 0, "dislikes": 0},
    {"id": 147, "name": "宇智波佐助", "movie": "火影忍者", "likes": 0, "dislikes": 0},
    {"id": 148, "name": "卡卡西", "movie": "火影忍者", "likes": 0, "dislikes": 0},
    {"id": 149, "name": "春野樱", "movie": "火影忍者", "likes": 0, "dislikes": 0},
    {"id": 150, "name": "雏田", "movie": "火影忍者", "likes": 0, "dislikes": 0},
    {"id": 151, "name": "五条悟", "movie": "咒术回战", "likes": 0, "dislikes": 0},
    {"id": 152, "name": "虎杖悠仁", "movie": "咒术回战", "likes": 0, "dislikes": 0},
    {"id": 153, "name": "江户川柯南", "movie": "名侦探柯南", "likes": 0, "dislikes": 0},
    {"id": 154, "name": "怪盗基德", "movie": "名侦探柯南", "likes": 0, "dislikes": 0},
    {"id": 155, "name": "毛利兰", "movie": "名侦探柯南", "likes": 0, "dislikes": 0},
    {"id": 156, "name": "灰原哀", "movie": "名侦探柯南", "likes": 0, "dislikes": 0},
    {"id": 157, "name": "樱木花道", "movie": "灌篮高手", "likes": 0, "dislikes": 0},
    {"id": 158, "name": "流川枫", "movie": "灌篮高手", "likes": 0, "dislikes": 0},
    {"id": 159, "name": "三井寿", "movie": "灌篮高手", "likes": 0, "dislikes": 0},
    {"id": 160, "name": "宫城良田", "movie": "灌篮高手", "likes": 0, "dislikes": 0},
    {"id": 161, "name": "赤木刚宪", "movie": "灌篮高手", "likes": 0, "dislikes": 0},
    {"id": 162, "name": "哆啦A梦", "movie": "哆啦A梦", "likes": 0, "dislikes": 0},
    {"id": 163, "name": "野比大雄", "movie": "哆啦A梦", "likes": 0, "dislikes": 0},
    {"id": 164, "name": "蜡笔小新", "movie": "蜡笔小新", "likes": 0, "dislikes": 0},
    {"id": 165, "name": "孙悟空", "movie": "大圣归来", "likes": 0, "dislikes": 0},
    {"id": 166, "name": "江流儿", "movie": "大圣归来", "likes": 0, "dislikes": 0},
    {"id": 167, "name": "李白", "movie": "长安三万里", "likes": 0, "dislikes": 0},
    {"id": 168, "name": "高适", "movie": "长安三万里", "likes": 0, "dislikes": 0},
    {"id": 169, "name": "罗小黑", "movie": "罗小黑战记", "likes": 0, "dislikes": 0},
    {"id": 170, "name": "无限", "movie": "罗小黑战记", "likes": 0, "dislikes": 0},
    {"id": 171, "name": "深津一成", "movie": "灌篮高手", "likes": 0, "dislikes": 0},
    {"id": 172, "name": "泽北荣治", "movie": "灌篮高手", "likes": 0, "dislikes": 0},

    # ==========================================
    # 11. 迪士尼/皮克斯 动画
    # ==========================================
    {"id": 173, "name": "艾莎 (Elsa)", "movie": "冰雪奇缘", "likes": 0, "dislikes": 0},
    {"id": 174, "name": "安娜 (Anna)", "movie": "冰雪奇缘", "likes": 0, "dislikes": 0},
    {"id": 175, "name": "雪宝", "movie": "冰雪奇缘", "likes": 0, "dislikes": 0},
    {"id": 176, "name": "朱迪", "movie": "疯狂动物城", "likes": 0, "dislikes": 0},
    {"id": 177, "name": "尼克", "movie": "疯狂动物城", "likes": 0, "dislikes": 0},
    {"id": 178, "name": "闪电 (树懒)", "movie": "疯狂动物城", "likes": 0, "dislikes": 0},
    {"id": 179, "name": "焦焦 (焦虑)", "movie": "头脑特工队2", "likes": 0, "dislikes": 0},
    {"id": 180, "name": "乐乐 (快乐)", "movie": "头脑特工队2", "likes": 0, "dislikes": 0},
    {"id": 181, "name": "忧忧 (忧伤)", "movie": "头脑特工队2", "likes": 0, "dislikes": 0},
    {"id": 182, "name": "怒怒", "movie": "头脑特工队2", "likes": 0, "dislikes": 0},
    {"id": 183, "name": "辛巴", "movie": "狮子王", "likes": 0, "dislikes": 0},
    {"id": 184, "name": "彭彭", "movie": "狮子王", "likes": 0, "dislikes": 0},
    {"id": 185, "name": "丁满", "movie": "狮子王", "likes": 0, "dislikes": 0},
    {"id": 186, "name": "米格", "movie": "寻梦环游记", "likes": 0, "dislikes": 0},
    {"id": 187, "name": "埃克托", "movie": "寻梦环游记", "likes": 0, "dislikes": 0},
    {"id": 188, "name": "瓦力", "movie": "机器人总动员", "likes": 0, "dislikes": 0},
    {"id": 189, "name": "伊娃", "movie": "机器人总动员", "likes": 0, "dislikes": 0},
    {"id": 190, "name": "胡迪", "movie": "玩具总动员", "likes": 0, "dislikes": 0},
    {"id": 191, "name": "巴斯光年", "movie": "玩具总动员", "likes": 0, "dislikes": 0},
    {"id": 192, "name": "大白 (Baymax)", "movie": "超能陆战队", "likes": 0, "dislikes": 0},
    {"id": 193, "name": "小宏", "movie": "超能陆战队", "likes": 0, "dislikes": 0},
    {"id": 194, "name": "毛怪 (苏利文)", "movie": "怪兽电力公司", "likes": 0, "dislikes": 0},
    {"id": 195, "name": "大眼仔", "movie": "怪兽电力公司", "likes": 0, "dislikes": 0},
    {"id": 196, "name": "尼莫", "movie": "海底总动员", "likes": 0, "dislikes": 0},
    {"id": 197, "name": "多莉", "movie": "海底总动员", "likes": 0, "dislikes": 0},
    {"id": 198, "name": "卡尔爷爷", "movie": "飞屋环游记", "likes": 0, "dislikes": 0},
    {"id": 199, "name": "小罗", "movie": "飞屋环游记", "likes": 0, "dislikes": 0},

    # ==========================================
    # 12. 经典电影角色 (填充至300+)
    # ==========================================
    {"id": 200, "name": "杰克", "movie": "泰坦尼克号", "likes": 0, "dislikes": 0},
    {"id": 201, "name": "露丝", "movie": "泰坦尼克号", "likes": 0, "dislikes": 0},
    {"id": 202, "name": "安迪", "movie": "肖申克的救赎", "likes": 0, "dislikes": 0},
    {"id": 203, "name": "瑞德", "movie": "肖申克的救赎", "likes": 0, "dislikes": 0},
    {"id": 204, "name": "程蝶衣", "movie": "霸王别姬", "likes": 0, "dislikes": 0},
    {"id": 205, "name": "段小楼", "movie": "霸王别姬", "likes": 0, "dislikes": 0},
    {"id": 206, "name": "1900", "movie": "海上钢琴师", "likes": 0, "dislikes": 0},
    {"id": 207, "name": "阿甘", "movie": "阿甘正传", "likes": 0, "dislikes": 0},
    {"id": 208, "name": "珍妮", "movie": "阿甘正传", "likes": 0, "dislikes": 0},
    {"id": 209, "name": "里昂", "movie": "这个杀手不太冷", "likes": 0, "dislikes": 0},
    {"id": 210, "name": "玛蒂尔达", "movie": "这个杀手不太冷", "likes": 0, "dislikes": 0},
    {"id": 211, "name": "楚门", "movie": "楚门的世界", "likes": 0, "dislikes": 0},
    {"id": 212, "name": "至尊宝", "movie": "大话西游", "likes": 0, "dislikes": 0},
    {"id": 213, "name": "紫霞仙子", "movie": "大话西游", "likes": 0, "dislikes": 0},
    {"id": 214, "name": "唐僧 (罗家英)", "movie": "大话西游", "likes": 0, "dislikes": 0},
    {"id": 215, "name": "八公", "movie": "忠犬八公的故事", "likes": 0, "dislikes": 0},
    {"id": 216, "name": "杰克·萨利", "movie": "阿凡达", "likes": 0, "dislikes": 0},
    {"id": 217, "name": "奈蒂莉", "movie": "阿凡达", "likes": 0, "dislikes": 0},
    {"id": 218, "name": "擎天柱", "movie": "变形金刚", "likes": 0, "dislikes": 0},
    {"id": 219, "name": "大黄蜂", "movie": "变形金刚", "likes": 0, "dislikes": 0},
    {"id": 220, "name": "威震天", "movie": "变形金刚", "likes": 0, "dislikes": 0},
    {"id": 221, "name": "哥斯拉", "movie": "哥斯拉大战金刚", "likes": 0, "dislikes": 0},
    {"id": 222, "name": "金刚", "movie": "哥斯拉大战金刚", "likes": 0, "dislikes": 0},
    {"id": 223, "name": "杰瑞 (Jerry)", "movie": "猫和老鼠", "likes": 0, "dislikes": 0},
    {"id": 224, "name": "汤姆 (Tom)", "movie": "猫和老鼠", "likes": 0, "dislikes": 0},
    {"id": 225, "name": "海绵宝宝", "movie": "海绵宝宝", "likes": 0, "dislikes": 0},
    {"id": 226, "name": "派大星", "movie": "海绵宝宝", "likes": 0, "dislikes": 0},
    {"id": 227, "name": "章鱼哥", "movie": "海绵宝宝", "likes": 0, "dislikes": 0},
    {"id": 228, "name": "蟹老板", "movie": "海绵宝宝", "likes": 0, "dislikes": 0},
    {"id": 229, "name": "马里奥", "movie": "超级马里奥大电影", "likes": 0, "dislikes": 0},
    {"id": 230, "name": "路易吉", "movie": "超级马里奥大电影", "likes": 0, "dislikes": 0},
    {"id": 231, "name": "碧琪公主", "movie": "超级马里奥大电影", "likes": 0, "dislikes": 0},
    {"id": 232, "name": "库巴", "movie": "超级马里奥大电影", "likes": 0, "dislikes": 0},
    {"id": 233, "name": "佛罗多", "movie": "指环王", "likes": 0, "dislikes": 0},
    {"id": 234, "name": "甘道夫", "movie": "指环王", "likes": 0, "dislikes": 0},
    {"id": 235, "name": "阿拉贡", "movie": "指环王", "likes": 0, "dislikes": 0},
    {"id": 236, "name": "莱戈拉斯", "movie": "指环王", "likes": 0, "dislikes": 0},
    {"id": 237, "name": "咕噜", "movie": "指环王", "likes": 0, "dislikes": 0},
    {"id": 238, "name": "尼奥", "movie": "黑客帝国", "likes": 0, "dislikes": 0},
    {"id": 239, "name": "崔妮蒂", "movie": "黑客帝国", "likes": 0, "dislikes": 0},
    {"id": 240, "name": "教父", "movie": "教父", "likes": 0, "dislikes": 0},
    {"id": 241, "name": "多姆 (唐老大)", "movie": "速度与激情", "likes": 0, "dislikes": 0},
    {"id": 242, "name": "布莱恩", "movie": "速度与激情", "likes": 0, "dislikes": 0},
    {"id": 243, "name": "约翰·威克", "movie": "疾速追杀", "likes": 0, "dislikes": 0},
    {"id": 244, "name": "伊桑·亨特", "movie": "碟中谍", "likes": 0, "dislikes": 0},
    {"id": 245, "name": "詹姆斯·邦德", "movie": "007系列", "likes": 0, "dislikes": 0},
    {"id": 246, "name": "杰克船长", "movie": "加勒比海盗", "likes": 0, "dislikes": 0},
    {"id": 247, "name": "威尔·特纳", "movie": "加勒比海盗", "likes": 0, "dislikes": 0},
    {"id": 248, "name": "伊丽莎白", "movie": "加勒比海盗", "likes": 0, "dislikes": 0},
    {"id": 249, "name": "程勇", "movie": "我不是药神", "likes": 0, "dislikes": 0},
    {"id": 250, "name": "吕受益", "movie": "我不是药神", "likes": 0, "dislikes": 0},
    {"id": 251, "name": "冷锋", "movie": "战狼2", "likes": 0, "dislikes": 0},
    {"id": 252, "name": "刘培强", "movie": "流浪地球", "likes": 0, "dislikes": 0},
    {"id": 253, "name": "图恒宇", "movie": "流浪地球2", "likes": 0, "dislikes": 0},
    {"id": 254, "name": "周喆直", "movie": "流浪地球2", "likes": 0, "dislikes": 0},
    {"id": 255, "name": "张麻子", "movie": "让子弹飞", "likes": 0, "dislikes": 0},
    {"id": 256, "name": "汤师爷", "movie": "让子弹飞", "likes": 0, "dislikes": 0},
    {"id": 257, "name": "黄四郎", "movie": "让子弹飞", "likes": 0, "dislikes": 0},
    {"id": 258, "name": "夏洛", "movie": "夏洛特烦恼", "likes": 0, "dislikes": 0},
    {"id": 259, "name": "马冬梅", "movie": "夏洛特烦恼", "likes": 0, "dislikes": 0},
    {"id": 260, "name": "袁华", "movie": "夏洛特烦恼", "likes": 0, "dislikes": 0},
    {"id": 261, "name": "陈念", "movie": "少年的你", "likes": 0, "dislikes": 0},
    {"id": 262, "name": "小北", "movie": "少年的你", "likes": 0, "dislikes": 0},
    {"id": 263, "name": "王多鱼", "movie": "西虹市首富", "likes": 0, "dislikes": 0},
    {"id": 264, "name": "李诗情", "movie": "开端 (剧场版)", "likes": 0, "dislikes": 0},
    {"id": 265, "name": "肖鹤云", "movie": "开端 (剧场版)", "likes": 0, "dislikes": 0},
    {"id": 266, "name": "范闲", "movie": "庆余年", "likes": 0, "dislikes": 0},
    {"id": 267, "name": "陈萍萍", "movie": "庆余年", "likes": 0, "dislikes": 0},
    {"id": 268, "name": "庆帝", "movie": "庆余年", "likes": 0, "dislikes": 0},
    {"id": 269, "name": "梅长苏", "movie": "琅琊榜", "likes": 0, "dislikes": 0},
    {"id": 270, "name": "靖王", "movie": "琅琊榜", "likes": 0, "dislikes": 0},
    {"id": 271, "name": "飞流", "movie": "琅琊榜", "likes": 0, "dislikes": 0},
    {"id": 272, "name": "明兰", "movie": "知否知否", "likes": 0, "dislikes": 0},
    {"id": 273, "name": "顾廷烨", "movie": "知否知否", "likes": 0, "dislikes": 0},
    {"id": 274, "name": "魏无羡", "movie": "陈情令", "likes": 0, "dislikes": 0},
    {"id": 275, "name": "蓝忘机", "movie": "陈情令", "likes": 0, "dislikes": 0},
    {"id": 276, "name": "甄嬛", "movie": "甄嬛传", "likes": 0, "dislikes": 0},
    {"id": 277, "name": "华妃", "movie": "甄嬛传", "likes": 0, "dislikes": 0},
    {"id": 278, "name": "皇后", "movie": "甄嬛传", "likes": 0, "dislikes": 0},
    {"id": 279, "name": "果郡王", "movie": "甄嬛传", "likes": 0, "dislikes": 0},
    {"id": 280, "name": "如懿", "movie": "如懿传", "likes": 0, "dislikes": 0},
    {"id": 281, "name": "东方不败", "movie": "笑傲江湖", "likes": 0, "dislikes": 0},
    {"id": 282, "name": "令狐冲", "movie": "笑傲江湖", "likes": 0, "dislikes": 0},
    {"id": 283, "name": "小龙女", "movie": "神雕侠侣", "likes": 0, "dislikes": 0},
    {"id": 284, "name": "杨过", "movie": "神雕侠侣", "likes": 0, "dislikes": 0},
    {"id": 285, "name": "赵敏", "movie": "倚天屠龙记", "likes": 0, "dislikes": 0},
    {"id": 286, "name": "张无忌", "movie": "倚天屠龙记", "likes": 0, "dislikes": 0},
    {"id": 287, "name": "周芷若", "movie": "倚天屠龙记", "likes": 0, "dislikes": 0},
    {"id": 288, "name": "乔峰", "movie": "天龙八部", "likes": 0, "dislikes": 0},
    {"id": 289, "name": "段誉", "movie": "天龙八部", "likes": 0, "dislikes": 0},
    {"id": 290, "name": "虚竹", "movie": "天龙八部", "likes": 0, "dislikes": 0},
    {"id": 291, "name": "金克丝", "movie": "英雄联盟：双城之战", "likes": 0, "dislikes": 0},
    {"id": 292, "name": "蔚 (Vi)", "movie": "英雄联盟：双城之战", "likes": 0, "dislikes": 0},
    {"id": 293, "name": "凯特琳", "movie": "英雄联盟：双城之战", "likes": 0, "dislikes": 0},
    {"id": 294, "name": "杰斯", "movie": "英雄联盟：双城之战", "likes": 0, "dislikes": 0},
    {"id": 295, "name": "维克托", "movie": "英雄联盟：双城之战", "likes": 0, "dislikes": 0},
    {"id": 296, "name": "大卫·马丁内斯", "movie": "赛博朋克：边缘行者", "likes": 0, "dislikes": 0},
    {"id": 297, "name": "露西", "movie": "赛博朋克：边缘行者", "likes": 0, "dislikes": 0},
    {"id": 298, "name": "瑞贝卡", "movie": "赛博朋克：边缘行者", "likes": 0, "dislikes": 0},
    {"id": 299, "name": "芙莉莲", "movie": "葬送的芙莉莲", "likes": 0, "dislikes": 0},
    {"id": 300, "name": "费伦", "movie": "葬送的芙莉莲", "likes": 0, "dislikes": 0},
    {"id": 301, "name": "阿尼亚", "movie": "间谍过家家", "likes": 0, "dislikes": 0},
    {"id": 302, "name": "劳德 (黄昏)", "movie": "间谍过家家", "likes": 0, "dislikes": 0},
    {"id": 303, "name": "约尔", "movie": "间谍过家家", "likes": 0, "dislikes": 0},
    {"id": 304, "name": "露比 (Ruby)", "movie": "我推的孩子", "likes": 0, "dislikes": 0},
    {"id": 305, "name": "星野爱", "movie": "我推的孩子", "likes": 0, "dislikes": 0},
    {"id": 306, "name": "阿奎亚", "movie": "我推的孩子", "likes": 0, "dislikes": 0},
    {"id": 307, "name": "电锯人 (淀治)", "movie": "电锯人", "likes": 0, "dislikes": 0},
    {"id": 308, "name": "玛奇玛", "movie": "电锯人", "likes": 0, "dislikes": 0},
    {"id": 309, "name": "帕瓦", "movie": "电锯人", "likes": 0, "dislikes": 0},
    {"id": 310, "name": "早川秋", "movie": "电锯人", "likes": 0, "dislikes": 0},
    {"id": 311, "name": "三笠", "movie": "进击的巨人", "likes": 0, "dislikes": 0},
    {"id": 312, "name": "艾伦", "movie": "进击的巨人", "likes": 0, "dislikes": 0},
    {"id": 313, "name": "利威尔 (兵长)", "movie": "进击的巨人", "likes": 0, "dislikes": 0},
    {"id": 314, "name": "阿尔敏", "movie": "进击的巨人", "likes": 0, "dislikes": 0},
    {"id": 315, "name": "吉克", "movie": "进击的巨人", "likes": 0, "dislikes": 0},
    {"id": 316, "name": "光头强爸爸", "movie": "熊出没 (系列)", "likes": 0, "dislikes": 0},
    {"id": 317, "name": "喜羊羊父母", "movie": "喜羊羊 (系列)", "likes": 0, "dislikes": 0},
    {"id": 318, "name": "包包大人", "movie": "喜羊羊 (系列)", "likes": 0, "dislikes": 0},
    {"id": 319, "name": "泰哥", "movie": "喜羊羊 (系列)", "likes": 0, "dislikes": 0},
    {"id": 320, "name": "扁嘴伦", "movie": "喜羊羊 (系列)", "likes": 0, "dislikes": 0},
]

# ==========================================
# ⭐️ 投票系统数据持久化 (JSON文件版) ⭐️
# ==========================================

# 1. 定义数据文件的存储路径 (存放在项目根目录下)
VOTE_DB_FILE = os.path.join(settings.BASE_DIR, 'vote_counts.json')


# 2. 辅助函数：读取硬盘上的票数
def load_vote_counts():
    if not os.path.exists(VOTE_DB_FILE):
        return {}  # 如果文件不存在，返回空字典
    try:
        with open(VOTE_DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}  # 如果文件坏了，返回空字典


# 3. 辅助函数：把票数写回硬盘
def save_vote_counts(data):
    try:
        with open(VOTE_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存投票数据失败: {e}")


# 4. 原始人物列表 (静态信息)
CHARACTER_DATA_STATIC = [
    # ... 这里请保留你原本那长长的 300 多号人物列表 ...
    # ==========================================
    # 4. 原始人物列表 (完整版 - 修复丢失数据)
    # ==========================================
    # 1. 2025年 1-2月
    {"id": 1, "name": "郭靖 (肖战饰)", "movie": "射雕英雄传：侠之大者 (2025)", "likes": 0, "dislikes": 0},
    {"id": 2, "name": "黄蓉 (庄达菲饰)", "movie": "射雕英雄传：侠之大者 (2025)", "likes": 0, "dislikes": 0},
    {"id": 3, "name": "哪吒", "movie": "哪吒之魔童闹海 (2025)", "likes": 0, "dislikes": 0},
    {"id": 4, "name": "敖丙", "movie": "哪吒之魔童闹海 (2025)", "likes": 0, "dislikes": 0},
    {"id": 5, "name": "姬发", "movie": "封神第二部：战火西岐 (2025)", "likes": 0, "dislikes": 0},
    {"id": 6, "name": "殷寿", "movie": "封神第二部：战火西岐 (2025)", "likes": 0, "dislikes": 0},
    {"id": 7, "name": "姜子牙", "movie": "封神第二部：战火西岐 (2025)", "likes": 0, "dislikes": 0},
    {"id": 8, "name": "邓婵玉", "movie": "封神第二部：战火西岐 (2025)", "likes": 0, "dislikes": 0},
    {"id": 9, "name": "唐仁", "movie": "唐探1900 (2025)", "likes": 0, "dislikes": 0},
    {"id": 10, "name": "秦风", "movie": "唐探1900 (2025)", "likes": 0, "dislikes": 0},
    {"id": 11, "name": "崔文子", "movie": "蛟龙行动 (2025)", "likes": 0, "dislikes": 0},
    {"id": 12, "name": "高豪", "movie": "蛟龙行动 (2025)", "likes": 0, "dislikes": 0},
    {"id": 13, "name": "帕丁顿熊", "movie": "柏灵顿：熊熊去秘鲁 (2025)", "likes": 0, "dislikes": 0},
    {"id": 14, "name": "布朗先生", "movie": "柏灵顿：熊熊去秘鲁 (2025)", "likes": 0, "dislikes": 0},
    {"id": 15, "name": "米奇", "movie": "米奇17 (2025)", "likes": 0, "dislikes": 0},
    {"id": 16, "name": "美国队长 (山姆)", "movie": "美国队长4 (2025)", "likes": 0, "dislikes": 0},

    # 2. 熊出没宇宙
    {"id": 17, "name": "熊大", "movie": "熊出没·重启未来 (2025)", "likes": 0, "dislikes": 0},
    {"id": 18, "name": "熊二", "movie": "熊出没·重启未来 (2025)", "likes": 0, "dislikes": 0},
    {"id": 19, "name": "光头强", "movie": "熊出没·重启未来 (2025)", "likes": 0, "dislikes": 0},
    {"id": 20, "name": "吉吉国王", "movie": "熊出没·重返地球", "likes": 0, "dislikes": 0},
    {"id": 21, "name": "毛毛", "movie": "熊出没·重返地球", "likes": 0, "dislikes": 0},
    {"id": 22, "name": "蹦蹦", "movie": "熊出没之雪岭熊风", "likes": 0, "dislikes": 0},
    {"id": 23, "name": "涂涂 (猫头鹰)", "movie": "熊出没之雪岭熊风", "likes": 0, "dislikes": 0},
    {"id": 24, "name": "团子 (山神)", "movie": "熊出没之雪岭熊风", "likes": 0, "dislikes": 0},
    {"id": 25, "name": "纳雅", "movie": "熊出没·奇幻空间", "likes": 0, "dislikes": 0},
    {"id": 26, "name": "乐天", "movie": "熊出没·狂野大陆", "likes": 0, "dislikes": 0},
    {"id": 27, "name": "小可", "movie": "熊出没·狂野大陆", "likes": 0, "dislikes": 0},
    {"id": 28, "name": "阿布", "movie": "熊出没·重返地球", "likes": 0, "dislikes": 0},
    {"id": 29, "name": "赵琳", "movie": "熊出没·探险日记", "likes": 0, "dislikes": 0},
    {"id": 30, "name": "天才威", "movie": "熊出没·怪兽计划", "likes": 0, "dislikes": 0},
    {"id": 31, "name": "李老板", "movie": "熊出没 (系列)", "likes": 0, "dislikes": 0},
    {"id": 32, "name": "肥波", "movie": "熊出没 (系列)", "likes": 0, "dislikes": 0},

    # 3. 喜羊羊与灰太狼宇宙
    {"id": 33, "name": "喜羊羊", "movie": "喜羊羊之守护 (2024)", "likes": 0, "dislikes": 0},
    {"id": 34, "name": "灰太狼", "movie": "喜羊羊之守护 (2024)", "likes": 0, "dislikes": 0},
    {"id": 35, "name": "美羊羊", "movie": "喜羊羊之筐出未来", "likes": 0, "dislikes": 0},
    {"id": 36, "name": "懒羊羊", "movie": "喜羊羊之筐出未来", "likes": 0, "dislikes": 0},
    {"id": 37, "name": "沸羊羊", "movie": "喜羊羊之筐出未来", "likes": 0, "dislikes": 0},
    {"id": 38, "name": "暖羊羊", "movie": "喜羊羊之筐出未来", "likes": 0, "dislikes": 0},
    {"id": 39, "name": "慢羊羊", "movie": "喜羊羊之筐出未来", "likes": 0, "dislikes": 0},
    {"id": 40, "name": "红太狼", "movie": "喜羊羊之筐出未来", "likes": 0, "dislikes": 0},
    {"id": 41, "name": "小灰灰", "movie": "喜羊羊之筐出未来", "likes": 0, "dislikes": 0},
    {"id": 42, "name": "蕉太狼", "movie": "喜羊羊与灰太狼", "likes": 0, "dislikes": 0},
    {"id": 43, "name": "机械羊", "movie": "喜羊羊之羊羊运动会", "likes": 0, "dislikes": 0},
    {"id": 44, "name": "潇洒哥", "movie": "喜羊羊之古古怪界", "likes": 0, "dislikes": 0},
    {"id": 45, "name": "黑大帅", "movie": "喜羊羊之古古怪界", "likes": 0, "dislikes": 0},
    {"id": 46, "name": "虎威太岁", "movie": "喜羊羊之虎虎生威", "likes": 0, "dislikes": 0},
    {"id": 47, "name": "小乖乖", "movie": "喜羊羊之牛气冲天", "likes": 0, "dislikes": 0},
    {"id": 48, "name": "智羊羊", "movie": "喜羊羊之跨时空救兵", "likes": 0, "dislikes": 0},

    # 4. 2024年 7-8月
    {"id": 49, "name": "马继业", "movie": "抓娃娃", "likes": 0, "dislikes": 0},
    {"id": 50, "name": "马成钢", "movie": "抓娃娃", "likes": 0, "dislikes": 0},
    {"id": 51, "name": "春兰", "movie": "抓娃娃", "likes": 0, "dislikes": 0},
    {"id": 52, "name": "死侍 (韦德)", "movie": "死侍与金刚", "likes": 0, "dislikes": 0},
    {"id": 53, "name": "金刚狼 (罗根)", "movie": "死侍与金刚", "likes": 0, "dislikes": 0},
    {"id": 54, "name": "小白", "movie": "白蛇：浮生", "likes": 0, "dislikes": 0},
    {"id": 55, "name": "许仙", "movie": "白蛇：浮生", "likes": 0, "dislikes": 0},
    {"id": 56, "name": "小青", "movie": "白蛇：浮生", "likes": 0, "dislikes": 0},
    {"id": 57, "name": "雷恩 (Rain)", "movie": "异形：夺命舰", "likes": 0, "dislikes": 0},
    {"id": 58, "name": "安迪 (Andy)", "movie": "异形：夺命舰", "likes": 0, "dislikes": 0},
    {"id": 59, "name": "高志垒", "movie": "逆行人生", "likes": 0, "dislikes": 0},
    {"id": 60, "name": "容金珍", "movie": "解密", "likes": 0, "dislikes": 0},
    {"id": 61, "name": "张楚岚", "movie": "异人之下", "likes": 0, "dislikes": 0},
    {"id": 62, "name": "冯宝宝", "movie": "异人之下", "likes": 0, "dislikes": 0},
    {"id": 63, "name": "格鲁", "movie": "神偷奶爸4", "likes": 0, "dislikes": 0},
    {"id": 64, "name": "小黄人", "movie": "神偷奶爸4", "likes": 0, "dislikes": 0},

    # 5. 2024年 1月
    {"id": 65, "name": "杜乐莹", "movie": "热辣滚烫", "likes": 0, "dislikes": 0},
    {"id": 66, "name": "昊坤", "movie": "热辣滚烫", "likes": 0, "dislikes": 0},
    {"id": 67, "name": "张驰", "movie": "飞驰人生2", "likes": 0, "dislikes": 0},
    {"id": 68, "name": "厉小海", "movie": "飞驰人生2", "likes": 0, "dislikes": 0},
    {"id": 69, "name": "孙宇强", "movie": "飞驰人生2", "likes": 0, "dislikes": 0},
    {"id": 70, "name": "韩明", "movie": "第二十条", "likes": 0, "dislikes": 0},
    {"id": 71, "name": "李茂娟", "movie": "第二十条", "likes": 0, "dislikes": 0},
    {"id": 72, "name": "程程", "movie": "金手指", "likes": 0, "dislikes": 0},
    {"id": 73, "name": "庄辉", "movie": "年会不能停！", "likes": 0, "dislikes": 0},
    {"id": 74, "name": "胡建林", "movie": "年会不能停！", "likes": 0, "dislikes": 0},
    {"id": 75, "name": "林阵安", "movie": "潜行", "likes": 0, "dislikes": 0},
    {"id": 76, "name": "修浩", "movie": "潜行", "likes": 0, "dislikes": 0},

    # 6. 2025年 7-8月
    {"id": 77, "name": "超人 (克拉克)", "movie": "超人 (2025)", "likes": 0, "dislikes": 0},
    {"id": 78, "name": "路易丝", "movie": "超人 (2025)", "likes": 0, "dislikes": 0},
    {"id": 79, "name": "神奇先生", "movie": "神奇四侠 (2025)", "likes": 0, "dislikes": 0},
    {"id": 80, "name": "隐形女侠", "movie": "神奇四侠 (2025)", "likes": 0, "dislikes": 0},
    {"id": 81, "name": "霹雳火", "movie": "神奇四侠 (2025)", "likes": 0, "dislikes": 0},
    {"id": 82, "name": "石头人", "movie": "神奇四侠 (2025)", "likes": 0, "dislikes": 0},
    {"id": 83, "name": "大坏狼", "movie": "坏蛋联盟2", "likes": 0, "dislikes": 0},
    {"id": 84, "name": "灶门炭治郎", "movie": "鬼灭之刃: 无限城篇", "likes": 0, "dislikes": 0},
    {"id": 85, "name": "鬼舞辻无惨", "movie": "鬼灭之刃: 无限城篇", "likes": 0, "dislikes": 0},
    {"id": 86, "name": "李善德", "movie": "长安的荔枝", "likes": 0, "dislikes": 0},
    {"id": 87, "name": "佐拉", "movie": "侏罗纪世界: 重生", "likes": 0, "dislikes": 0},
    {"id": 88, "name": "杨贵妃", "movie": "长安的荔枝", "likes": 0, "dislikes": 0},

    # 7. 宫崎骏吉卜力
    {"id": 89, "name": "千寻", "movie": "千与千寻", "likes": 0, "dislikes": 0},
    {"id": 90, "name": "哈尔", "movie": "哈尔的移动城堡", "likes": 0, "dislikes": 0},
    {"id": 91, "name": "龙猫", "movie": "龙猫", "likes": 0, "dislikes": 0},
    {"id": 92, "name": "波妞", "movie": "悬崖上的金希儿", "likes": 0, "dislikes": 0},
    {"id": 93, "name": "无脸男", "movie": "千与千寻", "likes": 0, "dislikes": 0},
    {"id": 94, "name": "琪琪", "movie": "魔女宅急便", "likes": 0, "dislikes": 0},
    {"id": 95, "name": "吉吉 (黑猫)", "movie": "魔女宅急便", "likes": 0, "dislikes": 0},
    {"id": 96, "name": "珊 (幽灵公主)", "movie": "幽灵公主", "likes": 0, "dislikes": 0},
    {"id": 97, "name": "阿席达卡", "movie": "幽灵公主", "likes": 0, "dislikes": 0},
    {"id": 98, "name": "巴鲁", "movie": "天空之城", "likes": 0, "dislikes": 0},
    {"id": 99, "name": "希达", "movie": "天空之城", "likes": 0, "dislikes": 0},
    {"id": 100, "name": "宗介", "movie": "悬崖上的金希儿", "likes": 0, "dislikes": 0},
    {"id": 101, "name": "卡西法", "movie": "哈尔的移动城堡", "likes": 0, "dislikes": 0},
    {"id": 102, "name": "汤婆婆", "movie": "千与千寻", "likes": 0, "dislikes": 0},
    {"id": 103, "name": "白龙", "movie": "千与千寻", "likes": 0, "dislikes": 0},
    {"id": 104, "name": "小梅", "movie": "龙猫", "likes": 0, "dislikes": 0},
    {"id": 105, "name": "小月", "movie": "龙猫", "likes": 0, "dislikes": 0},
    {"id": 106, "name": "苍鹭", "movie": "你想活出怎样的人生", "likes": 0, "dislikes": 0},
    {"id": 107, "name": "牧真人", "movie": "你想活出怎样的人生", "likes": 0, "dislikes": 0},
    {"id": 108, "name": "红猪", "movie": "红猪", "likes": 0, "dislikes": 0},
    {"id": 109, "name": "娜乌西卡", "movie": "风之谷", "likes": 0, "dislikes": 0},

    # 8. 哈利波特魔法世界
    {"id": 110, "name": "哈利·波特", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 111, "name": "赫敏·格兰杰", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 112, "name": "罗恩·韦斯莱", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 113, "name": "邓布利多", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 114, "name": "斯内普", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 115, "name": "伏地魔", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 116, "name": "德拉科·马尔福", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 117, "name": "海格", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 118, "name": "小天狼星", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 119, "name": "多比", "movie": "哈利·波特", "likes": 0, "dislikes": 0},
    {"id": 120, "name": "纽特", "movie": "神奇动物在哪里", "likes": 0, "dislikes": 0},
    {"id": 121, "name": "嗅嗅", "movie": "神奇动物在哪里", "likes": 0, "dislikes": 0},

    # 9. 漫威/DC 超级英雄
    {"id": 122, "name": "钢铁侠", "movie": "复仇者联盟", "likes": 0, "dislikes": 0},
    {"id": 123, "name": "蜘蛛侠", "movie": "蜘蛛侠", "likes": 0, "dislikes": 0},
    {"id": 124, "name": "美国队长", "movie": "复仇者联盟", "likes": 0, "dislikes": 0},
    {"id": 125, "name": "雷神", "movie": "复仇者联盟", "likes": 0, "dislikes": 0},
    {"id": 126, "name": "洛基", "movie": "复仇者联盟", "likes": 0, "dislikes": 0},
    {"id": 127, "name": "黑寡妇", "movie": "复仇者联盟", "likes": 0, "dislikes": 0},
    {"id": 128, "name": "绿巨人", "movie": "复仇者联盟", "likes": 0, "dislikes": 0},
    {"id": 129, "name": "鹰眼", "movie": "复仇者联盟", "likes": 0, "dislikes": 0},
    {"id": 130, "name": "奇异博士", "movie": "奇异博士", "likes": 0, "dislikes": 0},
    {"id": 131, "name": "灭霸", "movie": "复仇者联盟", "likes": 0, "dislikes": 0},
    {"id": 132, "name": "毒液", "movie": "毒液", "likes": 0, "dislikes": 0},
    {"id": 133, "name": "蝙蝠侠", "movie": "蝙蝠侠", "likes": 0, "dislikes": 0},
    {"id": 134, "name": "小丑 (Joker)", "movie": "蝙蝠侠", "likes": 0, "dislikes": 0},
    {"id": 135, "name": "神奇女侠", "movie": "正义联盟", "likes": 0, "dislikes": 0},
    {"id": 136, "name": "海王", "movie": "海王", "likes": 0, "dislikes": 0},
    {"id": 137, "name": "闪电侠", "movie": "闪电侠", "likes": 0, "dislikes": 0},
    {"id": 138, "name": "哈莉·奎茵", "movie": "自杀小队", "likes": 0, "dislikes": 0},

    # 10. 日漫/国漫 热门
    {"id": 139, "name": "路飞", "movie": "海贼王", "likes": 0, "dislikes": 0},
    {"id": 140, "name": "索隆", "movie": "海贼王", "likes": 0, "dislikes": 0},
    {"id": 141, "name": "乔巴", "movie": "海贼王", "likes": 0, "dislikes": 0},
    {"id": 142, "name": "娜美", "movie": "海贼王", "likes": 0, "dislikes": 0},
    {"id": 143, "name": "山治", "movie": "海贼王", "likes": 0, "dislikes": 0},
    {"id": 144, "name": "艾斯", "movie": "海贼王", "likes": 0, "dislikes": 0},
    {"id": 145, "name": "红发香克斯", "movie": "海贼王", "likes": 0, "dislikes": 0},
    {"id": 146, "name": "漩涡鸣人", "movie": "火影忍者", "likes": 0, "dislikes": 0},
    {"id": 147, "name": "宇智波佐助", "movie": "火影忍者", "likes": 0, "dislikes": 0},
    {"id": 148, "name": "卡卡西", "movie": "火影忍者", "likes": 0, "dislikes": 0},
    {"id": 149, "name": "春野樱", "movie": "火影忍者", "likes": 0, "dislikes": 0},
    {"id": 150, "name": "雏田", "movie": "火影忍者", "likes": 0, "dislikes": 0},
    {"id": 151, "name": "五条悟", "movie": "咒术回战", "likes": 0, "dislikes": 0},
    {"id": 152, "name": "虎杖悠仁", "movie": "咒术回战", "likes": 0, "dislikes": 0},
    {"id": 153, "name": "江户川柯南", "movie": "名侦探柯南", "likes": 0, "dislikes": 0},
    {"id": 154, "name": "怪盗基德", "movie": "名侦探柯南", "likes": 0, "dislikes": 0},
    {"id": 155, "name": "毛利兰", "movie": "名侦探柯南", "likes": 0, "dislikes": 0},
    {"id": 156, "name": "灰原哀", "movie": "名侦探柯南", "likes": 0, "dislikes": 0},
    {"id": 157, "name": "樱木花道", "movie": "灌篮高手", "likes": 0, "dislikes": 0},
    {"id": 158, "name": "流川枫", "movie": "灌篮高手", "likes": 0, "dislikes": 0},
    {"id": 159, "name": "三井寿", "movie": "灌篮高手", "likes": 0, "dislikes": 0},
    {"id": 160, "name": "宫城良田", "movie": "灌篮高手", "likes": 0, "dislikes": 0},
    {"id": 161, "name": "赤木刚宪", "movie": "灌篮高手", "likes": 0, "dislikes": 0},
    {"id": 162, "name": "哆啦A梦", "movie": "哆啦A梦", "likes": 0, "dislikes": 0},
    {"id": 163, "name": "野比大雄", "movie": "哆啦A梦", "likes": 0, "dislikes": 0},
    {"id": 164, "name": "蜡笔小新", "movie": "蜡笔小新", "likes": 0, "dislikes": 0},
    {"id": 165, "name": "孙悟空", "movie": "大圣归来", "likes": 0, "dislikes": 0},
    {"id": 166, "name": "江流儿", "movie": "大圣归来", "likes": 0, "dislikes": 0},
    {"id": 167, "name": "李白", "movie": "长安三万里", "likes": 0, "dislikes": 0},
    {"id": 168, "name": "高适", "movie": "长安三万里", "likes": 0, "dislikes": 0},
    {"id": 169, "name": "罗小黑", "movie": "罗小黑战记", "likes": 0, "dislikes": 0},
    {"id": 170, "name": "无限", "movie": "罗小黑战记", "likes": 0, "dislikes": 0},
    {"id": 171, "name": "深津一成", "movie": "灌篮高手", "likes": 0, "dislikes": 0},
    {"id": 172, "name": "泽北荣治", "movie": "灌篮高手", "likes": 0, "dislikes": 0},

    # 11. 迪士尼/皮克斯 动画
    {"id": 173, "name": "艾莎 (Elsa)", "movie": "冰雪奇缘", "likes": 0, "dislikes": 0},
    {"id": 174, "name": "安娜 (Anna)", "movie": "冰雪奇缘", "likes": 0, "dislikes": 0},
    {"id": 175, "name": "雪宝", "movie": "冰雪奇缘", "likes": 0, "dislikes": 0},
    {"id": 176, "name": "朱迪", "movie": "疯狂动物城", "likes": 0, "dislikes": 0},
    {"id": 177, "name": "尼克", "movie": "疯狂动物城", "likes": 0, "dislikes": 0},
    {"id": 178, "name": "闪电 (树懒)", "movie": "疯狂动物城", "likes": 0, "dislikes": 0},
    {"id": 179, "name": "焦焦 (焦虑)", "movie": "头脑特工队2", "likes": 0, "dislikes": 0},
    {"id": 180, "name": "乐乐 (快乐)", "movie": "头脑特工队2", "likes": 0, "dislikes": 0},
    {"id": 181, "name": "忧忧 (忧伤)", "movie": "头脑特工队2", "likes": 0, "dislikes": 0},
    {"id": 182, "name": "怒怒", "movie": "头脑特工队2", "likes": 0, "dislikes": 0},
    {"id": 183, "name": "辛巴", "movie": "狮子王", "likes": 0, "dislikes": 0},
    {"id": 184, "name": "彭彭", "movie": "狮子王", "likes": 0, "dislikes": 0},
    {"id": 185, "name": "丁满", "movie": "狮子王", "likes": 0, "dislikes": 0},
    {"id": 186, "name": "米格", "movie": "寻梦环游记", "likes": 0, "dislikes": 0},
    {"id": 187, "name": "埃克托", "movie": "寻梦环游记", "likes": 0, "dislikes": 0},
    {"id": 188, "name": "瓦力", "movie": "机器人总动员", "likes": 0, "dislikes": 0},
    {"id": 189, "name": "伊娃", "movie": "机器人总动员", "likes": 0, "dislikes": 0},
    {"id": 190, "name": "胡迪", "movie": "玩具总动员", "likes": 0, "dislikes": 0},
    {"id": 191, "name": "巴斯光年", "movie": "玩具总动员", "likes": 0, "dislikes": 0},
    {"id": 192, "name": "大白 (Baymax)", "movie": "超能陆战队", "likes": 0, "dislikes": 0},
    {"id": 193, "name": "小宏", "movie": "超能陆战队", "likes": 0, "dislikes": 0},
    {"id": 194, "name": "毛怪 (苏利文)", "movie": "怪兽电力公司", "likes": 0, "dislikes": 0},
    {"id": 195, "name": "大眼仔", "movie": "怪兽电力公司", "likes": 0, "dislikes": 0},
    {"id": 196, "name": "尼莫", "movie": "海底总动员", "likes": 0, "dislikes": 0},
    {"id": 197, "name": "多莉", "movie": "海底总动员", "likes": 0, "dislikes": 0},
    {"id": 198, "name": "卡尔爷爷", "movie": "飞屋环游记", "likes": 0, "dislikes": 0},
    {"id": 199, "name": "小罗", "movie": "飞屋环游记", "likes": 0, "dislikes": 0},

    # 12. 经典电影角色
    {"id": 200, "name": "杰克", "movie": "泰坦尼克号", "likes": 0, "dislikes": 0},
    {"id": 201, "name": "露丝", "movie": "泰坦尼克号", "likes": 0, "dislikes": 0},
    {"id": 202, "name": "安迪", "movie": "肖申克的救赎", "likes": 0, "dislikes": 0},
    {"id": 203, "name": "瑞德", "movie": "肖申克的救赎", "likes": 0, "dislikes": 0},
    {"id": 204, "name": "程蝶衣", "movie": "霸王别姬", "likes": 0, "dislikes": 0},
    {"id": 205, "name": "段小楼", "movie": "霸王别姬", "likes": 0, "dislikes": 0},
    {"id": 206, "name": "1900", "movie": "海上钢琴师", "likes": 0, "dislikes": 0},
    {"id": 207, "name": "阿甘", "movie": "阿甘正传", "likes": 0, "dislikes": 0},
    {"id": 208, "name": "珍妮", "movie": "阿甘正传", "likes": 0, "dislikes": 0},
    {"id": 209, "name": "里昂", "movie": "这个杀手不太冷", "likes": 0, "dislikes": 0},
    {"id": 210, "name": "玛蒂尔达", "movie": "这个杀手不太冷", "likes": 0, "dislikes": 0},
    {"id": 211, "name": "楚门", "movie": "楚门的世界", "likes": 0, "dislikes": 0},
    {"id": 212, "name": "至尊宝", "movie": "大话西游", "likes": 0, "dislikes": 0},
    {"id": 213, "name": "紫霞仙子", "movie": "大话西游", "likes": 0, "dislikes": 0},
    {"id": 214, "name": "唐僧 (罗家英)", "movie": "大话西游", "likes": 0, "dislikes": 0},
    {"id": 215, "name": "八公", "movie": "忠犬八公的故事", "likes": 0, "dislikes": 0},
    {"id": 216, "name": "杰克·萨利", "movie": "阿凡达", "likes": 0, "dislikes": 0},
    {"id": 217, "name": "奈蒂莉", "movie": "阿凡达", "likes": 0, "dislikes": 0},
    {"id": 218, "name": "擎天柱", "movie": "变形金刚", "likes": 0, "dislikes": 0},
    {"id": 219, "name": "大黄蜂", "movie": "变形金刚", "likes": 0, "dislikes": 0},
    {"id": 220, "name": "威震天", "movie": "变形金刚", "likes": 0, "dislikes": 0},
    {"id": 221, "name": "哥斯拉", "movie": "哥斯拉大战金刚", "likes": 0, "dislikes": 0},
    {"id": 222, "name": "金刚", "movie": "哥斯拉大战金刚", "likes": 0, "dislikes": 0},
    {"id": 223, "name": "杰瑞 (Jerry)", "movie": "猫和老鼠", "likes": 0, "dislikes": 0},
    {"id": 224, "name": "汤姆 (Tom)", "movie": "猫和老鼠", "likes": 0, "dislikes": 0},
    {"id": 225, "name": "海绵宝宝", "movie": "海绵宝宝", "likes": 0, "dislikes": 0},
    {"id": 226, "name": "派大星", "movie": "海绵宝宝", "likes": 0, "dislikes": 0},
    {"id": 227, "name": "章鱼哥", "movie": "海绵宝宝", "likes": 0, "dislikes": 0},
    {"id": 228, "name": "蟹老板", "movie": "海绵宝宝", "likes": 0, "dislikes": 0},
    {"id": 229, "name": "马里奥", "movie": "超级马里奥大电影", "likes": 0, "dislikes": 0},
    {"id": 230, "name": "路易吉", "movie": "超级马里奥大电影", "likes": 0, "dislikes": 0},
    {"id": 231, "name": "碧琪公主", "movie": "超级马里奥大电影", "likes": 0, "dislikes": 0},
    {"id": 232, "name": "库巴", "movie": "超级马里奥大电影", "likes": 0, "dislikes": 0},
    {"id": 233, "name": "佛罗多", "movie": "指环王", "likes": 0, "dislikes": 0},
    {"id": 234, "name": "甘道夫", "movie": "指环王", "likes": 0, "dislikes": 0},
    {"id": 235, "name": "阿拉贡", "movie": "指环王", "likes": 0, "dislikes": 0},
    {"id": 236, "name": "莱戈拉斯", "movie": "指环王", "likes": 0, "dislikes": 0},
    {"id": 237, "name": "咕噜", "movie": "指环王", "likes": 0, "dislikes": 0},
    {"id": 238, "name": "尼奥", "movie": "黑客帝国", "likes": 0, "dislikes": 0},
    {"id": 239, "name": "崔妮蒂", "movie": "黑客帝国", "likes": 0, "dislikes": 0},
    {"id": 240, "name": "教父", "movie": "教父", "likes": 0, "dislikes": 0},
    {"id": 241, "name": "多姆 (唐老大)", "movie": "速度与激情", "likes": 0, "dislikes": 0},
    {"id": 242, "name": "布莱恩", "movie": "速度与激情", "likes": 0, "dislikes": 0},
    {"id": 243, "name": "约翰·威克", "movie": "疾速追杀", "likes": 0, "dislikes": 0},
    {"id": 244, "name": "伊桑·亨特", "movie": "碟中谍", "likes": 0, "dislikes": 0},
    {"id": 245, "name": "詹姆斯·邦德", "movie": "007系列", "likes": 0, "dislikes": 0},
    {"id": 246, "name": "杰克船长", "movie": "加勒比海盗", "likes": 0, "dislikes": 0},
    {"id": 247, "name": "威尔·特纳", "movie": "加勒比海盗", "likes": 0, "dislikes": 0},
    {"id": 248, "name": "伊丽莎白", "movie": "加勒比海盗", "likes": 0, "dislikes": 0},
    {"id": 249, "name": "程勇", "movie": "我不是药神", "likes": 0, "dislikes": 0},
    {"id": 250, "name": "吕受益", "movie": "我不是药神", "likes": 0, "dislikes": 0},
    {"id": 251, "name": "冷锋", "movie": "战狼2", "likes": 0, "dislikes": 0},
    {"id": 252, "name": "刘培强", "movie": "流浪地球", "likes": 0, "dislikes": 0},
    {"id": 253, "name": "图恒宇", "movie": "流浪地球2", "likes": 0, "dislikes": 0},
    {"id": 254, "name": "周喆直", "movie": "流浪地球2", "likes": 0, "dislikes": 0},
    {"id": 255, "name": "张麻子", "movie": "让子弹飞", "likes": 0, "dislikes": 0},
    {"id": 256, "name": "汤师爷", "movie": "让子弹飞", "likes": 0, "dislikes": 0},
    {"id": 257, "name": "黄四郎", "movie": "让子弹飞", "likes": 0, "dislikes": 0},
    {"id": 258, "name": "夏洛", "movie": "夏洛特烦恼", "likes": 0, "dislikes": 0},
    {"id": 259, "name": "马冬梅", "movie": "夏洛特烦恼", "likes": 0, "dislikes": 0},
    {"id": 260, "name": "袁华", "movie": "夏洛特烦恼", "likes": 0, "dislikes": 0},
    {"id": 261, "name": "陈念", "movie": "少年的你", "likes": 0, "dislikes": 0},
    {"id": 262, "name": "小北", "movie": "少年的你", "likes": 0, "dislikes": 0},
    {"id": 263, "name": "王多鱼", "movie": "西虹市首富", "likes": 0, "dislikes": 0},
    {"id": 264, "name": "李诗情", "movie": "开端 (剧场版)", "likes": 0, "dislikes": 0},
    {"id": 265, "name": "肖鹤云", "movie": "开端 (剧场版)", "likes": 0, "dislikes": 0},
    {"id": 266, "name": "范闲", "movie": "庆余年", "likes": 0, "dislikes": 0},
    {"id": 267, "name": "陈萍萍", "movie": "庆余年", "likes": 0, "dislikes": 0},
    {"id": 268, "name": "庆帝", "movie": "庆余年", "likes": 0, "dislikes": 0},
    {"id": 269, "name": "梅长苏", "movie": "琅琊榜", "likes": 0, "dislikes": 0},
    {"id": 270, "name": "靖王", "movie": "琅琊榜", "likes": 0, "dislikes": 0},
    {"id": 271, "name": "飞流", "movie": "琅琊榜", "likes": 0, "dislikes": 0},
    {"id": 272, "name": "明兰", "movie": "知否知否", "likes": 0, "dislikes": 0},
    {"id": 273, "name": "顾廷烨", "movie": "知否知否", "likes": 0, "dislikes": 0},
    {"id": 274, "name": "魏无羡", "movie": "陈情令", "likes": 0, "dislikes": 0},
    {"id": 275, "name": "蓝忘机", "movie": "陈情令", "likes": 0, "dislikes": 0},
    {"id": 276, "name": "甄嬛", "movie": "甄嬛传", "likes": 0, "dislikes": 0},
    {"id": 277, "name": "华妃", "movie": "甄嬛传", "likes": 0, "dislikes": 0},
    {"id": 278, "name": "皇后", "movie": "甄嬛传", "likes": 0, "dislikes": 0},
    {"id": 279, "name": "果郡王", "movie": "甄嬛传", "likes": 0, "dislikes": 0},
    {"id": 280, "name": "如懿", "movie": "如懿传", "likes": 0, "dislikes": 0},
    {"id": 281, "name": "东方不败", "movie": "笑傲江湖", "likes": 0, "dislikes": 0},
    {"id": 282, "name": "令狐冲", "movie": "笑傲江湖", "likes": 0, "dislikes": 0},
    {"id": 283, "name": "小龙女", "movie": "神雕侠侣", "likes": 0, "dislikes": 0},
    {"id": 284, "name": "杨过", "movie": "神雕侠侣", "likes": 0, "dislikes": 0},
    {"id": 285, "name": "赵敏", "movie": "倚天屠龙记", "likes": 0, "dislikes": 0},
    {"id": 286, "name": "张无忌", "movie": "倚天屠龙记", "likes": 0, "dislikes": 0},
    {"id": 287, "name": "周芷若", "movie": "倚天屠龙记", "likes": 0, "dislikes": 0},
    {"id": 288, "name": "乔峰", "movie": "天龙八部", "likes": 0, "dislikes": 0},
    {"id": 289, "name": "段誉", "movie": "天龙八部", "likes": 0, "dislikes": 0},
    {"id": 290, "name": "虚竹", "movie": "天龙八部", "likes": 0, "dislikes": 0},
    {"id": 291, "name": "金克丝", "movie": "英雄联盟：双城之战", "likes": 0, "dislikes": 0},
    {"id": 292, "name": "蔚 (Vi)", "movie": "英雄联盟：双城之战", "likes": 0, "dislikes": 0},
    {"id": 293, "name": "凯特琳", "movie": "英雄联盟：双城之战", "likes": 0, "dislikes": 0},
    {"id": 294, "name": "杰斯", "movie": "英雄联盟：双城之战", "likes": 0, "dislikes": 0},
    {"id": 295, "name": "维克托", "movie": "英雄联盟：双城之战", "likes": 0, "dislikes": 0},
    {"id": 296, "name": "大卫·马丁内斯", "movie": "赛博朋克：边缘行者", "likes": 0, "dislikes": 0},
    {"id": 297, "name": "露西", "movie": "赛博朋克：边缘行者", "likes": 0, "dislikes": 0},
    {"id": 298, "name": "瑞贝卡", "movie": "赛博朋克：边缘行者", "likes": 0, "dislikes": 0},
    {"id": 299, "name": "芙莉莲", "movie": "葬送的芙莉莲", "likes": 0, "dislikes": 0},
    {"id": 300, "name": "费伦", "movie": "葬送的芙莉莲", "likes": 0, "dislikes": 0},
    {"id": 301, "name": "阿尼亚", "movie": "间谍过家家", "likes": 0, "dislikes": 0},
    {"id": 302, "name": "劳德 (黄昏)", "movie": "间谍过家家", "likes": 0, "dislikes": 0},
    {"id": 303, "name": "约尔", "movie": "间谍过家家", "likes": 0, "dislikes": 0},
    {"id": 304, "name": "露比 (Ruby)", "movie": "我推的孩子", "likes": 0, "dislikes": 0},
    {"id": 305, "name": "星野爱", "movie": "我推的孩子", "likes": 0, "dislikes": 0},
    {"id": 306, "name": "阿奎亚", "movie": "我推的孩子", "likes": 0, "dislikes": 0},
    {"id": 307, "name": "电锯人 (淀治)", "movie": "电锯人", "likes": 0, "dislikes": 0},
    {"id": 308, "name": "玛奇玛", "movie": "电锯人", "likes": 0, "dislikes": 0},
    {"id": 309, "name": "帕瓦", "movie": "电锯人", "likes": 0, "dislikes": 0},
    {"id": 310, "name": "早川秋", "movie": "电锯人", "likes": 0, "dislikes": 0},
    {"id": 311, "name": "三笠", "movie": "进击的巨人", "likes": 0, "dislikes": 0},
    {"id": 312, "name": "艾伦", "movie": "进击的巨人", "likes": 0, "dislikes": 0},
    {"id": 313, "name": "利威尔 (兵长)", "movie": "进击的巨人", "likes": 0, "dislikes": 0},
    {"id": 314, "name": "阿尔敏", "movie": "进击的巨人", "likes": 0, "dislikes": 0},
    {"id": 315, "name": "吉克", "movie": "进击的巨人", "likes": 0, "dislikes": 0},
    {"id": 316, "name": "光头强爸爸", "movie": "熊出没 (系列)", "likes": 0, "dislikes": 0},
    {"id": 317, "name": "喜羊羊父母", "movie": "喜羊羊 (系列)", "likes": 0, "dislikes": 0},
    {"id": 318, "name": "包包大人", "movie": "喜羊羊 (系列)", "likes": 0, "dislikes": 0},
    {"id": 319, "name": "泰哥", "movie": "喜羊羊 (系列)", "likes": 0, "dislikes": 0},
    {"id": 320, "name": "扁嘴伦", "movie": "喜羊羊 (系列)", "likes": 0, "dislikes": 0}
]
    # ... (请务必确保你之前的完整列表在这里) ...


# 5. 核心逻辑：获取带实时票数的人物列表
def get_merged_character_data():
    # A. 读取硬盘上的最新票数
    saved_votes = load_vote_counts()  # 格式: {"1": {"likes": 10, "dislikes": 2}, ...}

    # B. 深拷贝静态列表，准备合并
    import copy
    merged_data = copy.deepcopy(CHARACTER_DATA_STATIC)
    """
    CHARACTER_DATA_STATIC 是全局变量（写死在代码顶部的）。
    如果我们直接在它上面改数据，下次再请求时数据可能就乱了。
    copy.deepcopy 相当于“复印”了一份点名册。我们在复印件上涂改，原件永远保持干净。
    """

    # C. 合并数据
    for char in merged_data:
        str_id = str(char['id'])  # JSON的键通常是字符串
        # 检查：这个人在草稿本(json文件上面）上有记录吗？
        if str_id in saved_votes:
            # 有的话，把票数抄到复印件上
            char['likes'] = saved_votes[str_id].get('likes', 0)
            char['dislikes'] = saved_votes[str_id].get('dislikes', 0)

    return merged_data


# --- API 1: 获取列表 (前端加载时调用) ---
@api_view(['GET'])
def get_vote_characters(request):
    # 1. 呼叫上面的辅助函数，拿到组装好的完整数据
    data = get_merged_character_data()
    # 2. 打包发给前端
    return JsonResponse(data, safe=False)


# --- API 2: 投票动作 (点击按钮时调用,前端点击爱心或鸡蛋后端就会执行这个代码) ---
@api_view(['POST'])
def vote_character(request):
    char_id = request.data.get('id')
    vote_type = request.data.get('type')  # 'like' 或 'dislike'

    if not char_id or vote_type not in ['like', 'dislike']:
        return JsonResponse({'status': 'error', 'msg': '无效参数'})

    # 1. 读取现有文件
    saved_votes = load_vote_counts()
    str_id = str(char_id)

    # 2. 如果这个ID还没有记录，初始化它
    if str_id not in saved_votes:
        saved_votes[str_id] = {'likes': 0, 'dislikes': 0}

    # 3. 增加票数
    if vote_type == 'like':
        saved_votes[str_id]['likes'] += 1
    else:
        saved_votes[str_id]['dislikes'] += 1

    # 4. ⭐️ 存回硬盘 (这是关键！永久保存)
    save_vote_counts(saved_votes)

    # 5. 返回最新票数给前端
    return JsonResponse({
        'status': 'success',
        'likes': saved_votes[str_id]['likes'],
        'dislikes': saved_votes[str_id]['dislikes']
    })


# ==============================================================================
# ⭐️ 全能数据看板 V2.0 (多页版) ⭐️
# ==============================================================================
# ==============================================================================
# ⭐️ 全能数据看板 V3.0 (大字体 + 无乱码修正版) ⭐️
# ==============================================================================
# ==============================================================================
# ⭐️ 全能数据看板 V4.0 (逻辑修正 + 字体超大版) ⭐️
# ==============================================================================
@api_view(['GET'])
def get_vote_statistics_chart(request):
    try:
        page = int(request.GET.get('page', 1))
    except ValueError:
        page = 1

    # 1. 拿到原始数据 (列表格式)
    data = get_merged_character_data()
    if not data:
        return Response({'status': 'error', 'msg': '无数据'})

    # 2. ⭐️ 变成表格 (DataFrame)
    #    pd.DataFrame：把杂乱的数据变成结构化的表格
    #    这就像是在内存里建立了一张 Excel 表，方便排序和计算。”
    df = pd.DataFrame(data)
    # 3. 排序 (算出谁是第一名)
    #    by='likes': 按点赞数排
    #    ascending=False: 降序 (最大的在前面)
    df['total_activity'] = df['likes'] + df['dislikes']
    #sort_values：自动排序，不用自己写复杂的冒泡排序算法
    df_sorted_likes = df.sort_values(by='likes', ascending=False)
    df_sorted_dislikes = df.sort_values(by='dislikes', ascending=False)

    # ⭐️ 画布设置：20x11 英寸，保证大屏清晰度
    # figsize=(20, 11): 这张纸很大 (20英寸宽，11英寸高)，保证在大屏幕上显示也很清晰。
    fig = plt.figure(figsize=(20, 11))
    # set_facecolor('#2e3b32'): 把背景涂成深绿色
    # 这里的颜色代码 #2e3b32 正好对应你前端网页的“黑板”颜色，为了视觉统一。
    fig.patch.set_facecolor('#2e3b32')

    # ⭐️ 全局大字体设置
    plt.rcParams['font.size'] = 16
    TITLE_SIZE = 26
    LABEL_SIZE = 18

    # ==========================
    # 📄 Page 1: 柱状图 (Bar)
    # ==========================
    if page == 1:
        # add_subplot(1, 2, 1):
        # 意思把画布切成 1行 2列，现在画第 1 个格子 (左边)
        ax1 = fig.add_subplot(1, 2, 1)
        # add_subplot(1, 2, 2):
        # 现在画第 2 个格子 (右边)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.set_facecolor('#2e3b32')
        ax2.set_facecolor('#2e3b32')

        # 左图：人气榜
        # 取出前 5 名
        top_likes = df_sorted_likes.head(5)
        # ⭐️ 渐变色魔法 (plt.cm.Reds)
        # 意思是从红色色谱里，切出 5 种不同深浅的红色。
        # 第一名最红，第五名比较淡。这样比单一颜色好看得多。
        colors1 = plt.cm.Reds(np.linspace(0.9, 0.4, 5))
        # ⭐️ 正式画柱子 (Bar Chart)
        # x轴是名字，y轴是票数
        bars1 = ax1.bar(top_likes['name'], top_likes['likes'], color=colors1)
        ax1.set_title('人气榜 TOP 5 (Likes)', color='#f1c40f', fontsize=TITLE_SIZE, fontweight='bold', pad=25)
        ax1.tick_params(axis='x', colors='white', labelsize=18, rotation=15)
        ax1.tick_params(axis='y', colors='white', labelsize=16)
        # 标注数值
        for bar in bars1:
            # bar.get_x() + bar.get_width() / 2 : 算出柱子的正中心位置 (X轴)
            # bar.get_height() : 算出柱子的高度 (Y轴)
            # text(...) : 在这个坐标写上具体的数字
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), int(bar.get_height()),# 要写的数字
                     ha='center', va='bottom', color='white', fontsize=20, fontweight='bold')

        # 右图：捣蛋榜
        #已经按“踩数”从高到低排好序的大表格。.head(5) 的意思就是取前 5 行”
        top_dislikes = df_sorted_dislikes.head(5)
        colors2 = plt.cm.Greys(np.linspace(0.9, 0.4, 5))
        """
        1.ax2：代表**“右边的画布”**（之前定义的 subplot(1, 2, 2)）。
        2.bar(...)：画柱子。
        3.X轴放名字 (name)。
        4.Y轴放数值 (dislikes)。
        5.颜色用刚才调好的灰色 (colors2)。
        """
        bars2 = ax2.bar(top_dislikes['name'], top_dislikes['dislikes'], color=colors2)
        ax2.set_title('踩蛋榜 TOP 5 (Dislikes)', color='#bdc3c7', fontsize=TITLE_SIZE, fontweight='bold', pad=25)
        ax2.tick_params(axis='x', colors='white', labelsize=18, rotation=15)
        ax2.tick_params(axis='y', colors='white', labelsize=16)
        for bar in bars2:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,# 1. 找 X 轴中心点
                bar.get_height(),# 2. 找 Y 轴最高点 (柱子顶端)
                int(bar.get_height()),# 3. 要写的文字 (具体的票数)
                ha='center', va='bottom',# 4. 对齐方式 (水平居中，垂直底部对齐)
                color='white', fontsize=20, fontweight='bold')# 5. 白色字体

    # ==========================
    # 📄 Page 2: 饼状图 (Pie) x 3
    # ==========================
    elif page == 2:
        # fig.add_subplot(1, 3, 1):
        # 意思：把整张纸切成 1行 3列，现在占用第 1 个位置 (左边)
        ax1 = fig.add_subplot(1, 3, 1)
        # 占用第 2 个位置 (中间)
        ax2 = fig.add_subplot(1, 3, 2)
        # 占用第 3 个位置 (右边)
        ax3 = fig.add_subplot(1, 3, 3)

        # 通用饼图函数 (字体已加大)
        """
        如果一个数据全是 0（比如没人投票），饼图没法画（0不能做除数）。所以如果是 0，就强制让大家平分，防止程序崩溃
        """
        def draw_pie(ax, labels, values, title, cmap):
            if sum(values) == 0: values = [1] * len(values)  # 防报错

            # 颜色生成
            if isinstance(cmap, list):
                colors = cmap  # 如果直接传了颜色列表
            else:
                # 情况B: 如果你传给我的是一个“色谱” (比如 Reds)，我就自动根据数据的数量生成深浅不一的颜色。
                colors = cmap(np.linspace(0.4, 0.9, len(values)))
            # 轻微炸裂,explode 是“爆炸/裂开”的意思。设为 0.05，是为了让饼图的每一块之间有一点点缝隙
            explode = [0.05] * len(values)

            wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',#自动计算百分比
                                              colors=colors, explode=explode, startangle=140,
                                              textprops={'color': "white", 'fontsize': 16, 'weight': 'bold'})
            ax.set_title(title, color='white', fontsize=22, fontweight='bold', pad=20)

        # --- 图1: 人气票仓 (前5 vs 其他) ---
        # 1. 掐头：取出前 5 名 (顶流)
        top_5_likes = df_sorted_likes.head(5)
        # 2. 去尾：算出第 6 名及以后所有人的票数总和 (长尾)
        # .iloc[5:] 意思是：从表格第 5 行开始切到最后一行
        others_likes = df_sorted_likes.iloc[5:]['likes'].sum()
        # 3. 喂给画饼神器
        draw_pie(ax1,
                 list(top_5_likes['name']) + ['其他角色'],
                 list(top_5_likes['likes']) + [others_likes],
                 '人气票仓分布 (Top 5 vs Other)', plt.cm.Reds)

        # --- 图2: 捣蛋票仓 (前5 vs 其他) ---
        top_5_dislikes = df_sorted_dislikes.head(5)
        others_dislikes = df_sorted_dislikes.iloc[5:]['dislikes'].sum()
        draw_pie(ax2,
                 list(top_5_dislikes['name']) + ['其他角色'],
                 list(top_5_dislikes['dislikes']) + [others_dislikes],
                 '踩蛋票仓分布 (Top 5 vs Other)', plt.cm.Greys)

        # --- 图3: 全站红黑大对决 (Likes vs Dislikes) ---
        # 统计全站所有 Likes 和 所有 Dislikes
        total_system_likes = df['likes'].sum()
        total_system_dislikes = df['dislikes'].sum()

        # 这是一个特殊的饼图，只有两部分
        draw_pie(ax3,
                 ['全站人气票 (Likes)', '全站踩蛋票 (Dislikes)'],
                 [total_system_likes, total_system_dislikes],
                 '全站流量性质占比',
                 ['#e74c3c', '#7f8c8d'])  # 直接指定 红 vs 灰

    # ==========================
    # 📄 Page 3: 折线图 (Line)
    # ==========================
    elif page == 3:
        # 1. 创建普通直角坐标系 (1行1列
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor('#2e3b32')

        # 2. 生成 X 轴：0 到 23 (代表24小时)
        x_hours = np.arange(0, 24)
        np.random.seed(42)
        # 模拟数据
        # 3. ⭐️ 核心算法：用“正态分布公式”模拟真实流量
        # np.exp(...) 是指数函数，画出来是一个“钟形曲线” (中间高两边低)。
        # -(x_hours - 20) ** 2 : 意思是高峰期在晚上 20点 (8 PM)。
        # random.randint : 加一点随机噪点，让线看起来不那么平滑，更像真实数据。
        trend_hero = 20 + 80 * np.exp(-(x_hours - 20) ** 2 / 10) + np.random.randint(0, 15, 24)
        trend_villain = 10 + 60 * np.exp(-(x_hours - 1) ** 2 / 5) + 60 * np.exp(
            -(x_hours - 23) ** 2 / 5) + np.random.randint(0, 10, 24)
        ## 4. 绘制两条线
        # 'o' 和 's' 是数据点的形状（圆点和方块）
        # fill_between : 在线下填满颜色，做成“面积图”的效果，更好看。
        ax.plot(x_hours, trend_hero, marker='o', color='#e74c3c', linewidth=5, label='人气榜首热度')
        ax.plot(x_hours, trend_villain, marker='s', color='#95a5a6', linewidth=4, linestyle='--', label='踩蛋榜首热度')

        ax.set_title('24小时访问热度趋势 (模拟)', color='#2ecc71', fontsize=TITLE_SIZE, fontweight='bold', pad=30)
        ax.set_xlabel('时间 (小时)', color='white', fontsize=20)
        ax.set_ylabel('实时票数/分钟 (模拟)', color='white', fontsize=20)
        ax.tick_params(axis='both', colors='white', labelsize=16)
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.legend(facecolor='#2e3b32', edgecolor='white', labelcolor='white', fontsize=18)
        ax.fill_between(x_hours, trend_hero, alpha=0.2, color='#e74c3c')

    # ==========================
    # 📄 Page 4: 雷达图 (Radar)
    # ==========================
    elif page == 4:
        # 1. ⭐️ 关键设置：极坐标系
        # polar=True : 告诉 Matplotlib “我要画圆的，不要画方的”。
        # 这一步就把直角坐标系变成了雷达图的基础。
        ax = fig.add_subplot(1, 1, 1, polar=True)
        ax.set_facecolor('#2e3b32')
        # 2. 选出两位“主角”
        hero = df_sorted_likes.iloc[0] # 人气王 (第一名)
        villain = df_sorted_dislikes.iloc[0]    # 踩蛋王 (第一名)

        labels = ['总热度', '粉丝粘性', '争议性', '爆发力', '传播度']
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        # 真实的计算逻辑 + 模拟的计算逻辑
        def calculate_stats(char):
            # 1. 真实：总热度 (满分100)
            heat = min((char['likes'] + char['dislikes']), 100)
            # 2. 半真实：粉丝粘性 (Likes 越多粘性越高)
            fans = min(char['likes'] * 1.5, 100)
            # 3. 真实：争议性 (黑票占比)+0.01 就是防止崩溃
            total = char['likes'] + char['dislikes'] + 0.001
            contra = (char['dislikes'] / total) * 100
            # 4. 模拟：爆发力 (随机)
            burst = np.random.randint(60, 95)
            # 5. 模拟：传播度 (随机)
            spread = np.random.randint(50, 90)
            return [heat, fans, contra, burst, spread]

        hero_stats = calculate_stats(hero)
        hero_stats += hero_stats[:1]
        villain_stats = calculate_stats(villain)
        villain_stats += villain_stats[:1]

        ax.plot(angles, hero_stats, color='#f1c40f', linewidth=4, label=f"人气王: {hero['name']}")
        ax.fill(angles, hero_stats, color='#f1c40f', alpha=0.25)

        ax.plot(angles, villain_stats, color='#9b59b6', linewidth=4, label=f"踩蛋王: {villain['name']}")
        ax.fill(angles, villain_stats, color='#9b59b6', alpha=0.25)

        ax.set_title('巅峰对决：五维战力雷达', color='#e056fd', fontsize=TITLE_SIZE, fontweight='bold', pad=40)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color='white', size=22, weight='bold')  # 标签特大
        ax.set_yticks([20, 60, 100])
        ax.set_yticklabels([])
        ax.spines['polar'].set_visible(False)
        ax.grid(color='white', alpha=0.2)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), facecolor='#2e3b32', edgecolor='white',
                  labelcolor='white', fontsize=18)

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)
    buffer.seek(0)
    graphic = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return Response({'status': 'success', 'image': 'data:image/png;base64,' + graphic})