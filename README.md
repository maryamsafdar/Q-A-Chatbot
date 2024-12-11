# AI Q&A Assistant

This project is an **AI-powered Q&A application** built with Streamlit, LangChain, OpenAI, and Pinecone. It allows users to upload PDF documents, query them for specific information, and receive context-based answers in real time. The app integrates a modern UI and leverages Pinecone for vector storage and efficient information retrieval.

---

## Features

- **PDF Upload**: Easily upload PDF documents for indexing and querying.
- **AI-Powered Queries**: Ask questions and get answers directly from the uploaded content using OpenAI's GPT models.
- **Pinecone Integration**: Efficient document embedding and retrieval using Pinecone vector storage.
- **Streamlit UI**: Clean and user-friendly interface for interaction.
- **Conversation History**: Maintains session-based chat history for a seamless experience.

---

## How to Run the Application

### 1. Prerequisites

Ensure the following tools and configurations are ready:

- **Python 3.8+** installed.
- API Keys:
  - OpenAI API key (`OPENAI_API_KEY`).
  - Pinecone API key (`PINECONE_API_KEY`).
- A Pinecone index created with an appropriate dimension (e.g., `1536` for OpenAI's `text-embedding-ada-002`).
- Python virtual environment (optional but recommended).

### 2. Clone the Repository

Clone this repository to your local machine:
```bash
git clone <repository-url>
cd <repository-name>
```
### 3. Set Up Environment Variables

Create a `.env` file in the project root directory and populate it with the following environment variables:

```env
OPENAI_API_KEY=<your_openai_api_key>
PINECONE_API_KEY=<your_pinecone_api_key>
PINECONE_API_ENVIRONMENT=<your_pinecone_environment>
PINECONE_INDEX_NAME=<your_index_name>
```

### 4. Install Dependencies

To ensure all necessary packages are installed, follow these steps:

1. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   ```
2. **Activate the Virtual Environment:**
   ```bash
   venv\Scripts\activate
   ```
3. **Install Required Packages:**
Use the requirements.txt file to install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```


### 5. Run the Application

Launch the Streamlit application:
```bash
   streamlit run main.py
```

This command will start a local web server and provide a URL (e.g., http://localhost:8501) to access the application.





