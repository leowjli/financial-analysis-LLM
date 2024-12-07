import streamlit as st
from stock_info import get_stock_info, filter_stocks, parallel_process_stocks, get_company_tickers, get_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import numpy as np
import os

load_dotenv()

# Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
os.environ['PINECONE_API_KEY'] = pinecone_api_key

index_name = "stocks"
namespace = "stock-descriptions"

hf_embeddings = HuggingFaceEmbeddings()
vectorstore = PineconeVectorStore(index_name=index_name, embedding=hf_embeddings)


def perform_rag(query):
    """
    Function to perform RAG and retrieve data context
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"),)
    pinecone_index = pc.Index(index_name)

    raw_query_embedding = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=10, include_metadata=True, namespace=namespace)
    top_matches

    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    print(augmented_query)

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY")
    )

    system_prompt = f"""You are an expert at providing answers about stocks. Please answer my question provided."""

    llm_response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    response = llm_response.choices[0].message.content
    return response


def main():
    st.title("Financial Analysis and Automation")

    st.header("Stock Information and Overview")

    st.sidebar.header("Find Company and Filter Information")
    query_prompt = st.sidebar.text_area("Search for a stock or company:", 
                                        "Enter your preferences:", 
                                        "e.g. I want to find companies that have a market cap > 1000 and volume > 10000")
    
    st.sidebar.markdown("""
    Search for stocks based on the Name, Ticker, City, State, Country, Market Cap, Sector, and more.
    """)

    # Display the dropdown items
    company_tickers = get_company_tickers()
    options = [stock['ticker'] for stock in company_tickers.values()]

    select_ticker = st.selectbox("Choose a company or stock to get more information", options)

    if select_ticker:
        st.subheader(f"Information on {select_ticker}")
        stock_info = get_stock_info(select_ticker)
        st.write(stock_info)

    mc_filter = st.sidebar.selectbox("Filter by Market Cap", ['>', '<', '>=', '<=', '='])
    mc_input_val = st.sidebar.number_input("Enter value for Market Cap", min_value=1, max_value=100000000)

    vol_filter = st.sidebar.selectbox("Filter by Volume", ['>', '<', '>=', '<=', '='])
    vol_input_val = st.sidebar.number_input("Enter value for Volume", min_value=1, max_value=100000000)

    if st.sidebar.button("Search"):
        if query_prompt:
            st.write(f"Searching for: {query_prompt}")

            filters = {
                "mc_operator": mc_filter,
                "mc_value": mc_input_val,
                "vol_operator": vol_filter,
                "vol_value": vol_input_val
            }

            matching_stocks = filter_stocks(query_prompt, company_tickers, filters)

            if matching_stocks:
                for stock in matching_stocks:
                    stock_info = get_stock_info(stock['ticker'])
                    st.write(f"**{stock['ticker']}**: {stock_info['Name']}")
                    st.write(f"Market Cap: {stock_info['Market Cap']}")
                    st.write(f"Volume: {stock_info['Volume']}")
                    st.write("---")
                
                # Prepare tickers
                tickers_to_process = [company_tickers[num]['ticker'] for num in company_tickers.keys()]

                # Process tickers
                parallel_process_stocks(tickers_to_process, query_prompt, max_workers=10)

                try:
                    rag_results = perform_rag(query_prompt)
                    st.subheader("Stock Overview")
                    st.write(rag_results)
                except Exception as e:
                    st.error(f"Error occurred while fetching information: {str(e)}")

            else:
                st.write("No matching stocks found based on your input and filters.")
        else:
            st.write("Please enter a query in the text box.")
