import base64
from bs4 import BeautifulSoup
import cv2
import fitz # pymupdf
from functools import lru_cache
import gradio as gr
import json
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import os
from PIL import Image
import requests
from typing import List


SYSTEM_PROMPT = "You are Kai, an exceptionally capable and versatile AI assistant made by Zen. You are provided with images, videos and texts as input, You should answer users query in Structured, Detailed and Better way, in Human Style. You are also Expert in every field and also learn and try to answer from contexts related to previous question. Try your best to give best response possible to user. You reply in detail like human, use short forms, structured format, friendly tone and emotions."
WEB_SYSTEM_PROMPT = "You are Kai, a helpful and very powerful web assistant made by Zen. You are provided with WEB results from which you can find informations to answer users query in Structured, Detailed and Better way, in Human Style. You are also Expert in every field and also learn and try to answer from contexts related to previous question. Try your best to give best response possible to user. You reply in detail like human, use short forms, structured format, friendly tone and emotions."
# 这个system Prompt离最新的Prompt太远了，中间隔了一个History，效果不行

EXAMPLES = [
    [
        {
            "text": "你好",
        }
    ],
    [
        {
            "text": "自我介绍一下吧",
        }
    ],
    [
        {
            "text": "lcm是什么？",
        }
    ],
    [
        {
            "text": "lcm是什么？",
            "files": ["example_files/lcm.pdf"]
        }
    ],
    [
        {
            "text": "纸上写了什么字？",
            "files": ["example_files/paper_with_text.png"]
        }
    ],
    [
        {
            "text": "视频中的超级英雄是谁？",
            "files": ["example_files/spiderman.gif"]
        }
    ],
    [
        {
            "text": "今天上海的天气怎么样？",
        }
    ],
]

image_extensions = Image.registered_extensions()
video_extensions = ("avi", "mp4", "mov", "mkv", "flv", "wmv", "mjpeg", "wav", "gif", "webm", "m4v", "3gp")

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

@lru_cache(maxsize=128)
def extract_text_from_webpage(html_content):
    """Extracts visible text from HTML content using BeautifulSoup."""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "form", "svg"]):
        tag.extract()
    visible_text = soup.get_text(strip=True)
    return visible_text

def web_search(query: str) -> str:
    """Performs a Baidu search and returns extracted text from the results as a single string."""
    term = query
    max_chars_per_page = 8000
    all_results = ""
    
    with requests.Session() as session:
        try:
            resp = session.get(
                url="https://www.baidu.com/s",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"},
                params={"wd": term, "num": 4},
                timeout=5,
                verify=False,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error during initial request: {e}")
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")
        result_block = soup.find_all("div", attrs={"class": "result"})

        for result in result_block:
            link = result.find("a", href=True)
            if link:
                link = link["href"]
                try:
                    webpage = session.get(link, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"}, timeout=5, verify=False)
                    webpage.raise_for_status()
                    visible_text = extract_text_from_webpage(webpage.text)
                    if len(visible_text) > max_chars_per_page:
                        visible_text = visible_text[:max_chars_per_page]
                    all_results += f"{link}\n{visible_text}\n\n"
                except requests.exceptions.RequestException as e:
                    all_results += f"{link}\nError fetching page: {e}\n\n"
    
    return all_results.strip()  # 去掉最后多余的换行符


def model_inference(user_prompt, history):
    if user_prompt["files"]:
        messages, model = file_handler(user_prompt, history)
    else:
        query = user_prompt["text"]
        func_caller = []
        functions_metadata = [
            {"type": "function", "function": {"name": "web_search", "description": "Search query on google and find latest information, info about any person, object, place thing, everything that available on google.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "web search query"}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "general_query", "description": "Reply general query of USER, with LLM like you. But it does not answer tough questions and latest info's.", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "A detailed prompt"}}, "required": ["prompt"]}}},
            {"type": "function", "function": {"name": "hard_query", "description": "Reply tough query of USER, using powerful LLM. But it does not answer latest info's.", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "A detailed prompt"}}, "required": ["prompt"]}}},
        ]
        func_caller.append({"role": "user", "content": f'[SYSTEM]You are a helpful assistant. You have access to the following functions: \n{str(functions_metadata)}\n\nTo use these functions respond with:\n{{ "name": "function_name", "arguments": {{ "arg_1": "value_1", "arg_1": "value_1", ... }} }} , Reply in JSOn format, you can call only one function at a time, So, choose functions wisely. [USER] {user_prompt["text"]}'})
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=func_caller,
        )
        response = response.choices[0].message.content
        function_name = json.loads(response)["name"]
        if function_name == "web_search":
            gr.Info("Searching Web")
            yield "Searching Web"
            
            results = web_search(query)
            
            gr.Info("Extracting relevant Info")
            yield "Extracting Relevant Info"
            
            # 定义一个递归字符文本拆分器，用于将文本分割成指定大小的块
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,  # 设定每个文本块的字符数
                chunk_overlap=32,  # 设定文本块之间的重叠字符数
                length_function=len,  # 使用len函数来计算文本长度
            )
            texts = text_splitter.split_text(results)  # 使用拆分器将提取的文本分割成多个部分
            docsearch = FAISS.from_texts(texts, DashScopeEmbeddings())

            query = user_prompt["text"]

            docs = docsearch.similarity_search(query)
            relvants = ""
            for i in range(len(docs)):
                relvants += docs[i].page_content

            messages = history_to_messages(history, WEB_SYSTEM_PROMPT)
            messages.append({"role": "user", "content": f"[USER] {query} ,  [WEB RESULTS] {relvants}"})
            model = "qwen-plus"
        else:
            # 如果没有输入文本或历史记录，初始化为空
            if history is None:
                history = []

            # 将历史记录转换为消息格式，并追加当前用户输入
            messages = history_to_messages(history, SYSTEM_PROMPT)
            messages.append({'role': "user", 'content': user_prompt["text"]})
            model = "qwen-plus"
    
    with open("messages.txt", "w") as file:
        file.write(str(messages))
    # 调用模型生成响应，流式输出
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    
    # 累积完整的响应内容
    full_response = ""
    for chunk in response:
        if chunk.choices:
            full_response += chunk.choices[0].delta.content
            yield full_response     


def file_handler(user_prompt, history):
    if history is None:
        history = []

    text_and_image = []
    for file in user_prompt["files"]:
        # print(file)
        # {'path': '/tmp/gradio/a40428c32a3ce02e71ae7bce2fb10eed4ffe5d1ff32bad98d633f831c128df1b/1.jpg', 'url': 'http://127.0.0.1:7869/file=/tmp/gradio/a40428c32a3ce02e71ae7bce2fb10eed4ffe5d1ff32bad98d633f831c128df1b/1.jpg', 'size': None, 'orig_name': '1.jpg', 'mime_type': 'image/jpeg', 'is_stream': False, 'meta': {'_type': 'gradio.FileData'}}
        if file["path"].endswith("pdf"):
            messages = history_to_messages(history, SYSTEM_PROMPT)    
            result = retrieval(user_prompt)
            messages.append({'role': "user", 'content': result})
            return messages, "qwen-plus"
        elif file["path"].endswith(video_extensions):
            # 如果文件是视频，处理视频输入
            messages = history_to_messages(history, SYSTEM_PROMPT)
            text_and_image.append({"type": "video", "video": video_to_base64_urls(file["path"])})
            if user_prompt["text"]:
                text_and_image.append({"type": "text", "text": user_prompt["text"]})
            else:
                text_and_image.append({"type": "text", "text": "分析一下现在上传的这个视频"})

            messages.append({'role': "user", 'content': text_and_image})
            return messages, "qwen-vl-max-latest"
        
        elif file["path"].endswith(tuple([i for i, f in image_extensions.items()])):
            # 如果文件是图片，处理图片输入
            messages = history_to_messages(history, SYSTEM_PROMPT)
            text_and_image.append({"type": "image_url", "image_url": {"url": image_to_base64_url(file["path"])}})
            text_and_image.append({"type": "text", "text": user_prompt["text"]})
            
            messages.append({'role': "user", 'content': text_and_image})
            return messages, "qwen-vl-max-latest"


def retrieval(user_prompt):
    # 初始化一个空字符串来存储提取的文本
    result = ''
    for file in user_prompt["files"]:
        doc = fitz.open(file["path"])
        # 遍历 PDF 文件中的每一页
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)  # 加载当前页
            text = page.get_text()  # 提取文本
            if text:  # 如果当前页提取到了文本
                result += text  # 将提取到的文本添加到结果字符串中
        # 关闭此 PDF 文件
        doc.close()

    # 定义一个递归字符文本拆分器，用于将文本分割成指定大小的块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # 设定每个文本块的字符数
        chunk_overlap=32,  # 设定文本块之间的重叠字符数
        length_function=len,  # 使用len函数来计算文本长度
    )
    texts = text_splitter.split_text(result)  # 使用拆分器将提取的文本分割成多个部分

    docsearch = FAISS.from_texts(texts, DashScopeEmbeddings())

    query = user_prompt["text"]

    docs = docsearch.similarity_search(query)
    relvants = ""
    for i in range(len(docs)):
        relvants += docs[i].page_content

    return relvants + "/n" +query


def history_to_messages(history, system_prompt):
    """
    将消息历史记录转换为模型可接受的消息格式。

    参数:
    - history: 用户与助手之间的消息历史记录列表。
    - system_prompt: 当前对话的系统提示，用于指导模型生成。

    返回:
    - 包含用户与助手消息的列表，按时间顺序排列。
    """
    messages = [{'role': 'system', 'content': system_prompt}]
    for h in history:
        # {'role': 'user', 'metadata': {'title': None}, 'content': FileMessage(file=FileData(path='/tmp/gradio/5693928f9416c62d60b5da87fef7d950bf68178743826b3c45d7dd6b1ee4f426/paper_with_text.png', url=None, size=None, orig_name=None, mime_type='image/png', is_stream=False, meta={'_type': 'gradio.FileData'}), alt_text=None)}
        if h["role"] == "user":
            if type(h['content']) is str:
                messages.append({'role': "user", 'content': h["content"]})
            else:
                messages.append({'role': "user", 'content': "用户上传了一个文件"})
        elif h["role"] == "assistant":
            messages.append({'role': "assistant", 'content': h["content"]})
    return messages


def image_to_base64_url(image):
    """
    动态创建Base64编码的图片URL字符串。

    参数:
    - image (str): 图片地址。

    返回:
    - str: 包含图片Base64编码的URL字符串。
    """
    # 获取文件的后缀名
    _, image_format = os.path.splitext(image)

    with open(image, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')

    # 根据图片格式确定MIME类型
    mime_type = f"image/{image_format.lstrip('.').lower()}"
    
    # 构建Base64图片URL
    image_url = f"data:{mime_type};base64,{base64_image}"
    
    return image_url


def video_to_base64_urls(video_path, frame_interval=30):
    """
    将视频文件转换为一系列Base64编码的图片URL。

    Args:
        video_path (str): 视频文件的路径。
        frame_interval (int): 提取帧的间隔，默认为30帧。

    Returns:
        list: 包含Base64编码图片URL的列表。
    """
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    base64_urls = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        # 只提取每隔 frame_interval 帧的图片，防止数据过大，大模型无法处理，同时加快了处理速度
        if frame_idx % frame_interval == 0:
            # 将 ndarray 转换为 jpg 格式的内存图像
            _, buffer = cv2.imencode('.jpg', frame)
            # 转换为字节流后进行 Base64 编码
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            # 返回Base64编码的图片URL字符串
            base64_url = f"data:image/jpeg;base64,{image_base64}"
            base64_urls.append(base64_url)

        frame_idx += 1

    cap.release()

    return base64_urls