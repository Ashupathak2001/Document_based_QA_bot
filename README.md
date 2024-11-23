# RAG QA Bot with FAISS

A Retrieval-Augmented Generation (RAG) based Question Answering system that uses FAISS for efficient similarity search and Cohere for answer generation. This project allows users to upload PDF documents and ask questions about their content through an interactive Streamlit interface.

## 📋 Features

- **PDF Processing**: Upload and process PDF documents of any size
- **Vector Search**: Efficient similarity search using FAISS vector database
- **Natural Language Generation**: Generate coherent answers using Cohere's language model
- **Interactive UI**: User-friendly Streamlit interface
- **Persistent Storage**: Save and load document embeddings between sessions
- **Context Visibility**: View relevant document contexts with similarity scores
- **Index Management**: Clear and rebuild document index as needed

## 🔧 Technical Architecture

- **Frontend**: Streamlit
- **Document Processing**: PyPDF2
- **Embedding Generation**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Storage**: FAISS (Facebook AI Similarity Search)
- **Answer Generation**: Cohere API
- **Data Persistence**: Local file system (JSON + FAISS index)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/rag-qa-bot.git
cd rag-qa-bot
```

2. **Create and activate virtual environment**
```bash
python -m venv venv

# On Windows
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:
```env
COHERE_API_KEY=your_cohere_api_key_here
```

## 📄 Project Structure
```
rag_qa_bot/
├── server.py             # Core RAG implementation
├── QA_bot.py             # Streamlit interface
├── requirements.txt   # Project dependencies
├── README.md         # Project documentation
├── .env              # Environment variables
└── data/             # Generated during runtime
    ├── faiss.index   # FAISS vector index
    └── chunks.json   # Document chunks
```

## 🚀 Usage

1. **Start the application**
```bash
streamlit run QA_bot.py
```

2. **Access the interface**
- Open your web browser
- Navigate to `http://localhost:8501`

3. **Using the QA Bot**
- Click "Upload a PDF document" to process a new document
- Type your question in the text input field
- View the generated answer and relevant contexts
- Use "Clear Index" to remove all processed documents

## 💡 Key Components

### DocumentProcessor
- Handles PDF text extraction
- Splits documents into manageable chunks
- Generates embeddings using Sentence Transformers

### FAISSIndex
- Manages vector storage and retrieval
- Provides efficient similarity search
- Handles index persistence

### RAGModel
- Coordinates document processing, storage, and retrieval
- Integrates with Cohere for answer generation
- Manages the complete QA pipeline

## ⚙️ Configuration

The system can be configured by modifying the following parameters:

- `chunk_size`: Maximum size of text chunks (default: 512 characters)
- `top_k`: Number of similar contexts to retrieve (default: 3)
- `max_tokens`: Maximum length of generated answers (default: 300)
- `temperature`: Creativity of answer generation (default: 0.7)

## 🌟 Example Usage

```python
from main import RAGModel

# Initialize the model
rag_model = RAGModel(cohere_api_key="your-cohere-api-key")

# Index a document
num_chunks = rag_model.index_document("path/to/document.pdf")

# Query the model
response = rag_model.query("What is discussed in the document?")
print("Answer:", response["answer"])
print("Contexts:", response["contexts"])
```

## 📊 Performance

- Embedding Dimension: 384 (all-MiniLM-L6-v2)
- Average Query Time: ~100ms
- Support for Multiple Documents
- Efficient Memory Usage with FAISS

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🚨 Troubleshooting

Common issues and solutions:

1. **PDF Processing Errors**
   - Ensure PDF is not password-protected
   - Check PDF file permissions
   - Verify PDF is not corrupted

2. **Memory Issues**
   - Reduce chunk size for large documents
   - Clear index periodically
   - Monitor system memory usage

3. **Generation Issues**
   - Verify Cohere API key
   - Check internet connection
   - Review API rate limits

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FAISS by Facebook Research
- Cohere API
- Sentence-Transformers
- Streamlit Team

## 📧 Contact

Your Name - ashupathak22@gmail.com
Project Link: https://github.com/Ashupathak2001/Document_based_QA_bot

## 🔄 Updates Log

- v1.0.0 (2024-03-23)
  - Initial release
  - Basic QA functionality
  - FAISS integration
  - Streamlit interface
