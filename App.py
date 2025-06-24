import os
from langgraph.prebuilt import create_react_agent 
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st
import re
import time
from dotenv import load_dotenv
import tempfile
import uuid

# Load environment variables
load_dotenv()

# Import your RAG tool from the other file
print(f"##### Main App Initialization #####")
try: 
    from Doc_QnA_RAG import rag_qa_tool, setup_rag_system
    print("Successfully imported the tools\n\n")
except ImportError as e:
    print(f"ERROR: Could not import a tool. Make sure the file is in the correct directory. {e}")
    st.error("Fatal Error: Could not load certain tools. Please check server logs.")
    st.stop()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


@tool
def doc_qna_tool(ques: str) -> str:
    """
    Use this tool when the user asks a question that requires retrieving information from an uploaded document or PDF.
    This tool can extract answers directly from the content of user-provided files.
    """
    return rag_qa_tool(ques)



system_message = SystemMessage(content="""You are a helpful assistant. 
Instructions:
1. Use doc_qna_tool when users ask about uploaded documents or PDFs
2. Always provide clear, helpful responses
5. If you use tools, integrate their outputs naturally into your response
6. Be conversational and professional

Remember: You have access to uploaded documents.""")

tools = [doc_qna_tool]

# Create the LangGraph agent
agent = create_react_agent(model=llm, tools=tools, prompt=system_message)
print("LangGraph ReAct agent created.")


def process_query(user_input, chat_history=None):
    """Process user query through the LangGraph agent"""
    print(f"\n--- Entering process_query ---")

    if chat_history is None:
        chat_history = []

    # ✅ Copy existing messages and append the new user input
    messages = chat_history.copy()
    messages.append(HumanMessage(content=user_input))

    print(f"Messages to be sent to the LLM: {len(messages)} messages.")

    try:
        # ✅ Invoke the agent with valid LangGraph message objects
        print("Invoking the agent with messages...")
        result = agent.invoke({"messages": messages})

        # ✅ Extract the final response
        final_response = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                final_response = msg.content
                break

        if final_response is None:
            final_response = "I apologize, but I couldn't generate a proper response."

        print(f"Extracted final_response: {final_response[:100]}...")

        # ✅ Return response + updated message history
        return final_response, result["messages"]

    except Exception as e:
        print(f"Error in process_query: {str(e)}")
        return f"I encountered an error: {str(e)}", messages

def main():
    st.set_page_config(
        page_title="Doc-RAG", 
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .stChatMessage {
            margin-bottom: 1rem;
        }
        .main-header {
            padding: 1rem 0;
            border-bottom: 1px solid #e0e0e0;
            margin-bottom: 2rem;
        }
        .chat-container {
            max-height: 60vh;
            overflow-y: auto;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Personal RAG based assistant")
    st.caption("Your AI-powered assistant with document analysis")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_file" not in st.session_state:
        st.session_state.processed_file = None
    if "langgraph_messages" not in st.session_state:
        st.session_state.langgraph_messages = []

    # Sidebar for tools
    with st.sidebar:
        st.header("🛠️ Tools")
        
        # File upload
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
        
        if uploaded_file:
            if st.session_state.processed_file != uploaded_file.name:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                print(f"Saved uploaded file to: {temp_path}")

                with st.spinner("🔄 Processing document..."):
                    try:
                        setup_rag_system(temp_path)
                        st.success("✅ `Document processed successfully!")
                        st.session_state.processed_file = uploaded_file.name
                        print("setup_rag_system completed successfully.")
                    except Exception as e:
                        st.error(f"❌ Error processing document: {str(e)}")
                        print(f"Error in setup_rag_system: {e}")
            else:
                st.info(f"ℹ️ '{uploaded_file.name}' already processed!")
        
        # Show current file
        if st.session_state.processed_file:
            st.subheader("📄 Current Document")
            st.text(st.session_state.processed_file)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.langgraph_messages = []
            st.rerun()

    # Main chat area
    chat_container = st.container()
    
    # Display existing chat history
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about your document and I'll hopefully be able to answer is , LOL :) ..."):
        # Add user message to display history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Process query with LangGraph
                response_text, updated_langgraph_messages = process_query(
                    prompt, 
                    st.session_state.langgraph_messages
                )
                
                # Update LangGraph message history
                st.session_state.langgraph_messages = updated_langgraph_messages
                
                # Stream the response
                message_placeholder = st.empty()
                full_response = ""
                
                # Simulate streaming effect
                words = response_text.split()
                for word in words:
                    full_response += word + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.02)
                
                # Final response without cursor
                message_placeholder.markdown(full_response)
                
                # Add assistant response to display history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": full_response
                })

    # Welcome message for new users
    if not st.session_state.chat_history:
        st.markdown(""" Welcome!""")

if __name__ == "__main__":
    main()


























