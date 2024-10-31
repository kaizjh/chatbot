from bs4 import BeautifulSoup
from functools import lru_cache
from langchain.agents import tool
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI
import os
import requests
from typing import List

@lru_cache(maxsize=128)
def extract_text_from_webpage(html_content):
    """Extracts visible text from HTML content using BeautifulSoup."""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "form", "svg"]):
        tag.extract()
    visible_text = soup.get_text(strip=True)
    return visible_text

@tool
def web_search(query: str) -> List:
    """Performs a Baidu search and returns extracted text from the results."""
    term = query
    max_chars_per_page = 8000
    all_results = []
    
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
            return []

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
                    all_results.append({"link": link, "text": visible_text})
                except requests.exceptions.RequestException as e:
                    all_results.append({"link": link, "text": f"Error fetching page: {e}"})
    return all_results


tools = [web_search]

# 从消息创建聊天提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.",
        ),
        ("user", "{input}"),
    ]
)


llm = ChatOpenAI(
    model="qwen-max", 
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 将工具格式化为OpenAI函数并绑定到LLM
# 这个bind method应该来自from langchain_openai import ChatOpenAI，也就是说LangChain特有的method。
llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])


# 创建代理(定义或设置agent)
agent = (
    {
         # 定义输入的处理方式
        "input": lambda x: x["input"],       
        # 定义 agent_scratchpad 的处理方式，将中间步骤格式化为 OpenAI 函数消息
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

# 创建代理执行器实例，传入代理、工具和 verbose 参数设置为 True
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# 调用代理执行器的 invoke 方法，传入输入信息
# 这个executor.invoke不用print方法就可以在terminal中输出
agent_executor.invoke({"input": "上海今天的天气怎么样？"})




# {"type": "function", 
#  "function": 
#     {"name": "web_search", 
#      "description": "Search query on google and find latest information, info about any person, object, place thing, everything that available on google.", 
#      "parameters": 
#         {"type": "object", 
#          "properties": {"query": {"type": "string", "description": "web search query"}}, "required": ["query"]}}},
# {"type": "function", "function": {"name": "general_query", "description": "Reply general query of USER, with LLM like you. But it does not answer tough questions and latest info's.", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "A detailed prompt"}}, "required": ["prompt"]}}},
# {"type": "function", "function": {"name": "hard_query", "description": "Reply tough query of USER, using powerful LLM. But it does not answer latest info's.", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "A detailed prompt"}}, "required": ["prompt"]}}},
# Example usage
