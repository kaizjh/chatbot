from langchain.openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=2)
# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]


llm = ChatOpenAI(
    model="qwen-max", 
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])


# 创建代理(定义或设置agent)
agent = (
    {
         # 定义输入的处理方式
        "input": lambda x: x["input"],
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

# 创建代理执行器实例，传入代理、工具和 verbose 参数设置为 True
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

