export PYSPARK_PYTHON=/home/hadoop/anaconda/bin/python
export PYSPARK_DRIVER_PYTHON=/home/hadoop/anaconda/bin/python
pip install boto3

# Download NLTK libraries
/home/hadoop/anaconda/bin/python -c "import nltk; \
nltk.download('stopwords', 'nltk_data'); \
nltk.download('punkt', 'nltk_data'); \
nltk.download('wordnet', 'nltk_data'); \
nltk.download('averaged_perceptron_tagger', 'nltk_data'); \
nltk.download('maxent_treebank_pos_tagger', 'nltk_data')"
