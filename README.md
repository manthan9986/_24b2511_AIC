AI Community JE Assignment

NAME: P Manthan    DPT : MEMS    ROLL NO : 24B2511

---

**Setup steps for local execution  :** 

Setup steps for the first and second questions are very similar,following are the steps required to run the notebooks in local environment :

1.Install the python (if not installed already)

2.Place the notebook in a directory. 

3.Create a virtual environment and install the required libraries and some extra libraries along with the standard ones.

4.Launch the jupyter notebook and run the notebook 

For the third question since we are not judging the answer by its metrics and we are verifying the model by entering  a text prompt we can change the prompt in : 

response \= rag\_chatbot(“”)

**Detailed documentation of analysis,experiments and final comments   :** 

**1\.** Project focuses on classifying resumes into job categories using machine learning. The pipeline begins with loading and exploring the dataset, followed by stratified splitting to maintain balanced class distribution. Text preprocessing includes lowercase conversion, punctuation removal, and lemmatization using spaCy. For feature extraction, small BERT embeddings are generated in batches to handle memory constraints. The model architecture consists of a neural network with two hidden layers (100 units each) and employs focal loss to address class imbalance, along with class weighting. Training incorporates multiple metrics (accuracy, precision, recall, F1-score) over 30 epochs.

This project strengths include robust preprocessing, effective handling of class imbalance, and BERT's semantic understanding. Challenges like memory limits were addressed through batch processing and a smaller BERT variant. Potential improvements include experimenting with larger BERT models, alternative architectures, and enhanced text cleaning. The solution is modular, scalable, and suitable for deployment with minor refinements.

**2\.** This notebook demonstrates a transfer learning approach for classifying Fashion MNIST images using ResNet50. The workflow begins with preprocessing the 28x28 grayscale images by resizing them to 224x224 RGB format to match ResNet50's input requirements, followed by normalization and dataset splitting into training, validation, and test sets. The model leverages a pre-trained ResNet50 (with frozen weights) as a feature extractor, augmented with a global average pooling layer, a dense ReLU layer with dropout for regularization, and a final softmax layer for classification. The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss, then trained for 5 epochs with GPU acceleration. Key strengths include efficient data handling via TensorFlow's Dataset API and improved performance through transfer learning. Potential enhancements could involve fine-tuning the base model, implementing data augmentation, or experimenting with other architectures like EfficientNet. Overall, this implementation provides a robust and scalable solution for Fashion MNIST classification, combining deep learning best practices with transfer learning efficiency.

**3\.**This notebook demonstrates a Retrieval-Augmented Generation (RAG) system for extracting and answering questions from PDF documents. The workflow begins by loading and splitting PDF text into manageable chunks using PyPDF2 and LangChain's RecursiveCharacterTextSplitter, ensuring optimal context retention with overlapping segments. These chunks are then indexed using FAISS with MiniLM embeddings for efficient similarity searches. When a query is submitted, the system retrieves the most relevant document segments and generates accurate, context-bound answers using Groq's Llama3 API, which is constrained to only use the provided context for responses. A practical example shows the system successfully answering a question about technical details from an assignment PDF. Key strengths include fast retrieval with FAISS and precise, source-grounded answers through RAG. Potential enhancements could involve adding caching mechanisms, supporting additional file formats, or fine-tuning the LLM prompts for better performance. This system is ideal for document-based Q\&A, research assistance, or educational tools, offering a scalable solution for knowledge extraction from unstructured text.

**Error handling and troubleshooting   :** 

While working with TensorFlow and BERT models, several common issues may arise that can interrupt training or inference. One frequent error is related to incorrect slicing in pandas or NumPy, such as using `data[10000:0]`, which returns an empty result since Python slicing is exclusive of the end index and doesn't reverse by default. To reverse a sequence, use `[::-1]` or specify a negative step. Data type mismatches are another common issue—errors like `invalid dtype: object` usually happen when passing a Series of lists or arrays to a model instead of a 2D NumPy array. To fix this, stack the values using `np.stack()` and convert them to `float32`. Labels also need proper formatting: use `LabelEncoder` for integer encoding if using `"sparse_categorical_crossentropy"`, or `to_categorical()` for one-hot encoding with `"categorical_crossentropy"`. If you encounter deserialization or preprocessing issues with BERT, try re-downloading the model or switching to batch processing instead of using `.apply()`, which is inefficient. To train a model using BERT embeddings, ensure the input layer shape is `(512,)` and avoid Flatten layers if the data is already 1D. Class weights can be added during training via the `class_weight` argument in `.fit()`. For MacBook M3 users, GPU acceleration is available by installing `tensorflow-macos` and `tensorflow-metal`, which utilize Apple’s Metal API. Lastly, always check shapes and data types using `.shape`, `type()`, and `.dtype` before passing data to the model to avoid runtime errors. These practices ensure smoother development and easier debugging of deep learning pipelines.

The errors encountered were: **`KeyError: 'choices'`**, which occurs when the Groq API returns an unexpected response—usually due to an invalid or deprecated model. This can be fixed by switching to a supported model like `"llama3-8b-8192"` and adding a safeguard in the code to handle missing keys in the response. The second frequent issue was **`No secrets found`**, caused by using `st.secrets["GROQ_API_KEY"]` without creating a `.streamlit/secrets.toml` file. This can be resolved either by creating the secrets file with the API key or, for quick testing, directly hardcoding the API key into the script (not recommended for production).  
**References :**   
1\. Hands on Machine Learning with SickitLearn (by *Aurelien Geron*)  
2\. Lots of Medium articles  
3\. NLP bootcamp resources give during the first semester of the first year.

