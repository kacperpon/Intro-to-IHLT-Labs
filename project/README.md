# Semantic Textual Similarity Project

### Authors

- Kacper Poniatowski
- Pau Blanco

## Project Structure
project/
│
├── data/            
│   ├── test/
│   └── train.py              
├── src/                      
│   ├── __init__.py  
│   ├── data_analysis.py   
│   ├── feature_extraction.py   
│   ├── models.py   
│   ├── pipeline.ipynb   
│   ├── preprocessor.py   
│   └── utils.py              
├── README.md                 
└── requirements.txt          

## Setup Instructions
If you wish to run the cells within the pipeline notebook, follow these instructions:

1. Clone the repository or download the project files.

2. Create a virtual environment:
- **Windows**:
    ```bash
        python -m venv venv
    venv\Scripts\activate
    ```
- **macOS/Linux**:
    ```bash
    python3 -m venv venv
        source venv/bin/activate
    ```

3. Install the required dependencies:
    ```bash
        pip install -r requirements.txt
    ```
