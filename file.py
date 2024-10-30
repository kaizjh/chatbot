import base64
import cv2
import fitz # pymupdf
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from openai import OpenAI
from PIL import Image



# "You are Kai, an exceptionally capable and versatile AI assistant made by Zen. You are provided with images and texts as input, You should answer users query in Structured, Detailed and Better way, in Human Style. You are also Expert in every field and also learn and try to answer from contexts related to previous question. Try your best to give best response possible to user. You reply in detail like human, use short forms, structured format, friendly tone and emotions."
TEXT_SYSTEM_PROMPT = "你是Kai，一个由无名氏开发的全能的智能助手，你会尽全力帮助用户，用温和的语气来解答用户的疑问，也可以使用表情来使对话更加生动。"

IMAGE_SYSTEM_PROMPT = "你是Kai，一个由无名氏开发的善于分析图片的智能助手，你会尽全力帮助用户，用温和的语气来解答用户的疑问，也可以使用表情来使对话更加生动。"

VIDEO_SYSTEM_PROMPT = "你是Kai，一个由无名氏开发的善于分析视频的智能助手，你会尽全力帮助用户，用温和的语气来解答用户的疑问，也可以使用表情来使对话更加生动。"

PDF_SYSTEM_PROMPT = "你是Kai，一个由无名氏开发的善于分析文件的智能助手，你会尽全力帮助用户，用温和的语气来解答用户的疑问，也可以使用表情来使对话更加生动。"
# 这个system Prompt离最新的Prompt太远了，中间隔了一个History，效果不行
# "你是Kai，一个由无名氏开发的全能的智能助手，在最新的用户输入中会有一个视频，如果用户没有附上提问或者说明，你将会主动分析该视频；如果用户有针对该视频的提问或说明，你会尽全力帮助用户，用温和的语气来解答用户的疑问。也可以使用表情来使对话更加生动。"

image_extensions = Image.registered_extensions()
video_extensions = ("avi", "mp4", "mov", "mkv", "flv", "wmv", "mjpeg", "wav", "gif", "webm", "m4v", "3gp")

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def model_inference(user_prompt, history):
    if user_prompt["files"]:
        for chunk in image_and_video_handler(user_prompt, history):
            yield chunk
    else:
        # 如果没有输入文本或历史记录，初始化为空
        if history is None:
            history = []

        # 将历史记录转换为消息格式，并追加当前用户输入
        messages = history_to_messages(history, TEXT_SYSTEM_PROMPT)
        messages.append({'role': "user", 'content': user_prompt["text"]})
        # 调用模型生成响应，流式输出
        response = client.chat.completions.create(
            model="qwen-plus",
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


def image_and_video_handler(user_prompt, history):
    if history is None:
        history = []

    text_and_image = []
    print("-----------")
    print(user_prompt)
    for file in user_prompt["files"]:
        # print(file)
        # {'path': '/tmp/gradio/a40428c32a3ce02e71ae7bce2fb10eed4ffe5d1ff32bad98d633f831c128df1b/1.jpg', 'url': 'http://127.0.0.1:7869/file=/tmp/gradio/a40428c32a3ce02e71ae7bce2fb10eed4ffe5d1ff32bad98d633f831c128df1b/1.jpg', 'size': None, 'orig_name': '1.jpg', 'mime_type': 'image/jpeg', 'is_stream': False, 'meta': {'_type': 'gradio.FileData'}}
        if file["path"].endswith(video_extensions):
            # 如果文件是视频，处理视频输入
            messages = history_to_messages(history, VIDEO_SYSTEM_PROMPT)
            text_and_image.append({"type": "video", "video": file["url"]})
            if user_prompt["text"]:
                text_and_image.append({"type": "text", "text": user_prompt["text"]})
            else:
                text_and_image.append({"type": "text", "text": "分析一下现在上传的这个视频"})

            messages.append({'role': "user", 'content': text_and_image})
            yield call_llm(messages, "qwen-vl-max-latest")
        elif file["path"].endswith(tuple([i for i, f in image_extensions.items()])):
            # 如果文件是图片，处理图片输入
            messages = history_to_messages(history, IMAGE_SYSTEM_PROMPT)
            text_and_image.append({"type": "image_url", "image_url": {"url": file["url"]}})
            text_and_image.append({"type": "text", "text": user_prompt["text"]})
            
            messages.append({'role': "user", 'content': text_and_image})
            yield call_llm(messages, "qwen-vl-max-latest")
        elif file["path"].endswith("pdf"):
            messages = history_to_messages(history, PDF_SYSTEM_PROMPT)    
            
            result = retrieval(user_prompt)
            
            messages.append({'role': "user", 'content': result})
            yield call_llm(messages, "qwen-plus")


def call_llm(messages, model):
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
        if h["role"] == "user":
            if type(h['content']) == tuple:
                # 当用户上传图片或视频时，特殊处理此消息
                messages.append({'role': "user", 'content': "用户上传了一张图片或者视频"})
            else:
                messages.append({'role': "user", 'content': h["content"]})
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