import streamlit as st
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import pandas as pd


# Функция для отправки GET-запросов к GitHub API
def get_github_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Ошибка получения данных: {response.status_code}")
        return {}


# Функция для отображения топ-10 популярных репозиториев по звездам
def show_top_repositories(language_filter, stars_threshold):
    st.title("Популярные репозитории")

    url = f'https://api.github.com/search/repositories?q=stars:>{stars_threshold}+language:{language_filter}&sort=stars&order=desc&per_page=10'
    data = get_github_data(url)

    if 'items' not in data or not data['items']:
        st.error("Репозитории не найдены.")
        return

    repo_names = [repo['name'] for repo in data['items']]
    stars = [repo['stargazers_count'] for repo in data['items']]

    fig, ax = plt.subplots()
    sns.barplot(x=stars, y=repo_names, palette="viridis", ax=ax)
    ax.set_xlabel('Кол-во звезд')
    ax.set_ylabel('Название')
    ax.set_title('Топ репозиториев')

    st.pyplot(fig)


# Функция для отображения активности разработчиков
def show_developer_activity(language_filter):
    st.title("Активность разработчиков")

    url_pr = f'https://api.github.com/search/issues?q=type:pr+state:closed+is:public+language:{language_filter}&per_page=100'
    pr_data = get_github_data(url_pr)
    url_issue = f'https://api.github.com/search/issues?q=type:issue+state:closed+is:public+language:{language_filter}&per_page=100'
    issue_data = get_github_data(url_issue)

    if 'items' not in pr_data or 'items' not in issue_data:
        st.error("Данные о pr или issues не найдены.")
        return

    pr_counts = Counter([item['user']['login'] for item in pr_data['items']])
    issue_counts = Counter([item['user']['login'] for item in issue_data['items']])

    developers = pr_counts + issue_counts
    top_developers = developers.most_common(10)

    dev_names = [dev[0] for dev in top_developers]
    dev_counts = [dev[1] for dev in top_developers]

    fig, ax = plt.subplots()
    sns.barplot(x=dev_counts, y=dev_names, palette="rocket", ax=ax)
    ax.set_xlabel('Кол-во pr и issue')
    ax.set_ylabel('Разработчик')
    ax.set_title('Топ активных разработчиков')

    st.pyplot(fig)


# Функция для отображения трендов по языкам программирования
def show_language_trends(start_date, end_date, country_filter):
    st.title("Тренд технологий")

    # Получение репозиториев по стране
    url_country = f'https://api.github.com/search/repositories?q=created:{start_date}..{end_date}+user_location:{country_filter}&per_page=100'
    data = get_github_data(url_country)

    if 'items' not in data or not data['items']:
        st.error("Репозитории не найдены.")
        return

    languages = [repo['language'] for repo in data['items'] if repo['language'] is not None]

    language_counts = Counter(languages)
    labels, counts = zip(*language_counts.most_common(10))

    fig, ax = plt.subplots()
    ax.pie(counts, labels=labels, autopct='%1.1f%%')
    ax.axis('equal')
    ax.set_title(f'Разбиение по технологиям')

    st.pyplot(fig)


# Функция для отображения активности коммитов
def show_commit_activity():
    st.title("Временная активность по коммитам")

    url_commit = 'https://api.github.com/search/repositories?q=sort=stars&order=desc&per_page=100'
    data = get_github_data(url_commit)

    if 'items' not in data or not data['items']:
        st.error("Репозитории не найдены.")
        return

    commit_times = [datetime.strptime(repo['pushed_at'], "%Y-%m-%dT%H:%M:%SZ") for repo in data['items']]
    hours = [commit_time.hour for commit_time in commit_times]
    weekdays = [commit_time.weekday() for commit_time in commit_times]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    sns.histplot(hours, bins=24, ax=ax1, kde=True)
    ax1.set_title('Активность по часам дня')
    ax1.set_xlabel('Час дня')
    ax1.set_ylabel('Кол-во коммитов')

    sns.histplot(weekdays, bins=np.arange(8) - 0.5, ax=ax2, kde=True)
    ax2.set_title('Активность по дням недели')
    ax2.set_xlabel('День недели')
    ax2.set_ylabel('Кол-во коммитов')
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'])

    fig.tight_layout()

    st.pyplot(fig)


# Функция для отображения взаимосвязи между звездами и форками репозиториев и проверки гипотезы
def show_fork_vs_stars():
    st.title("Корреляционный анализ")

    url_fork = 'https://api.github.com/search/repositories?q=stars:>10000&per_page=100'
    data = get_github_data(url_fork)

    if 'items' not in data or not data['items']:
        st.error("Репозитории не найдены.")
        return

    stars = [repo['stargazers_count'] for repo in data['items']]
    forks = [repo['forks_count'] for repo in data['items']]
    issues = [repo['open_issues_count'] for repo in data['items']]

    fig, ax = plt.subplots()
    sns.scatterplot(x=stars, y=forks, alpha=0.5, ax=ax)
    ax.set_title('Взаимосвязь форков и звезд')
    ax.set_xlabel('Кол-во звезд')
    ax.set_ylabel('Кол-во форков')

    # Вычисление коэффициента корреляции и p
    correlation_coefficient, p_value = stats.pearsonr(stars, forks)
    st.write(f"Коэф. корреляции: {correlation_coefficient}")
    st.write(f"P-значение: {p_value}")

    if p_value < 0.05:
        st.write(
            "Есть основания полагать, что кол-во звезд и форков взаимосвязано.")
    else:
        st.write(
            "Нет оснований полагать, что кол-во звезд и форков взаимосвязано.")

    # Вычисление линейной регрессии
    slope, intercept, r_value, p_value, std_err = stats.linregress(stars, forks)
    line = slope * np.array(stars) + intercept

    sns.lineplot(x=stars, y=line, color='red', ax=ax, label=f'Корреляция: {correlation_coefficient:.2f}')

    ax.legend()

    st.pyplot(fig)


# Функция для выполнения кластерного анализа
def show_cluster_analysis():
    st.title("Кластерный анализ репозиториев")

    url_repo = 'https://api.github.com/search/repositories?q=stars:>1000&per_page=100'
    data = get_github_data(url_repo)

    if 'items' not in data or not data['items']:
        st.error("Репозитории не найдены.")
        return

    df = pd.DataFrame({
        'stars': [repo['stargazers_count'] for repo in data['items']],
        'forks': [repo['forks_count'] for repo in data['items']],
        'open_issues': [repo['open_issues_count'] for repo in data['items']]
    })
    kmeans = KMeans(n_clusters=3,  random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['stars', 'forks', 'open_issues']])
    cluster_names = {0: 'малоизвестные', 1: 'известные', 2: 'популярные'}
    df['cluster_name'] = df['cluster'].map(cluster_names)

    fig, ax = plt.subplots()
    sns.scatterplot(x=df['stars'], y=df['forks'], hue=df['cluster_name'], palette='viridis', ax=ax)
    ax.set_title('Кластерный анализ репозиториев')
    ax.set_xlabel('Кол-во звезд')
    ax.set_ylabel('Кол-во форков')

    st.pyplot(fig)

# Интерфейс Streamlit
st.sidebar.title("Анализ ОПО")

option = st.sidebar.selectbox(
    "Выберите:",
    ("Популярные репозитории", "Активность разработчиков", "Тренд технологий", "Временная активность по коммитам", "Корреляционный анализ", "Кластерный анализ репозиториев")
)

if option == "Популярные репозитории":
    language_filter = st.sidebar.text_input("Введите язык программирования:")
    stars_threshold = st.sidebar.number_input("Мин. кол-во звезд:", value=1000)
    show_top_repositories(language_filter, stars_threshold)
elif option == "Активность разработчиков":
    language_filter = st.sidebar.text_input("Введите язык программирования:")
    show_developer_activity(language_filter)
elif option == "Тренд технологий":
    start_date = st.sidebar.date_input("Начальная дата:", value=datetime(2023, 1, 1))
    end_date = st.sidebar.date_input("Конечная дата:", value=datetime.now(), max_value=datetime.now())
    country_filter = st.sidebar.text_input("Введите страну:")
    show_language_trends(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), country_filter)
elif option == "Временная активность по коммитам":
    show_commit_activity()
elif option == "Корреляционный анализ":
    show_fork_vs_stars()
elif option == "Кластерный анализ репозиториев":
    show_cluster_analysis()
