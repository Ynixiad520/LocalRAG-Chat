import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
import os

# --- 1. 配置界面 ---
st.set_page_config(page_title="DevMind - 程序员本地知识库", layout="wide")
st.title("💻 DevMind: 开发者本地技术助手")
st.markdown("---")

# --- 2. 初始化核心组件 ---
# 确保你已经安装并运行了 Ollama，且下载了 llama3 模型
EMBEDDING_MODEL = "nomic-embed-text" # 或者直接用 llama3 的 embedding
LLM_MODEL = "llama3"

# 侧边栏：设置知识库路径
with st.sidebar:
    st.header("配置中心")
    doc_path = st.text_input("输入你的技术文档文件夹路径", value="./my_docs")
    if st.button("更新/初始化知识库"):
        if os.path.exists(doc_path):
            with st.spinner("正在索引文档，请稍候..."):
                # 加载 Markdown 和 TXT
                loader = DirectoryLoader(doc_path, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
                docs = loader.load()
                
                # 切分文档
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                # 存储到向量数据库 (本地持久化)
                vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
                    persist_directory="./chroma_db"
                )
                st.success(f"成功索引了 {len(docs)} 份文档！")
        else:
            st.error("文件夹路径不存在，请检查。")

# --- 3. 聊天界面 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 输入框
if prompt := st.chat_input("向你的知识库提问 (例如：怎么实现 React 的性能优化？)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 检索并生成回答
    with st.chat_message("assistant"):
        try:
            # 加载现有的向量库
            vectorstore = Chroma(
                persist_directory="./chroma_db", 
                embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL)
            )
            
            # 创建问答链
            llm = ChatOllama(model=LLM_MODEL)
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={"verbose": True}
            )
            
            response = qa_chain.invoke(prompt)
            answer = response["result"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"出错了：请确保 Ollama 已启动并且已索引文档。错误信息：{e}")
