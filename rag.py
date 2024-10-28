import fitz # pymupdf
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def rag_inference(user_prompt, history):
    if history is None:
        history = []

    messages = history_to_messages(history, RAG_SYSTEM_PROMPT)

    files = user_prompt["files"]
    doc = fitz.open(files)
    # 初始化一个空字符串来存储提取的文本
    result = ''

    # 遍历 PDF 文件中的每一页
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # 加载当前页
        text = page.get_text()  # 提取文本
        if text:  # 如果当前页提取到了文本
            result += text  # 将提取到的文本添加到结果字符串中

    # 关闭 PDF 文件
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
    
    messages.append({'role': "user", 'content': relvants + "/n" +query})

    
    # 调用模型生成响应，流式输出
    response = client.chat.completions.create(
        model="qwen-vl-max-latest",
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
