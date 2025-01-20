# chunking-strategies

## Installation

- Ensure Python 3.10 is installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/).

### Steps

1. **Create a virtual environment:**

    ```sh
    python3.10 -m venv venv
    ```

2. **Activate the virtual environment:**

    - On macOS and Linux:

        ```sh
        source venv/bin/activate
        ```

    - On Windows:

        ```sh
        .\venv\Scripts\activate
        ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

This will set up a virtual environment using Python 3.10 and install all the dependencies listed in the `requirements.txt` file.

4. **Set up environment variables:**

    Create a `.env` file in the root directory of your project and add the following lines, replacing the placeholder values with your actual keys:

    ```dotenv
    # keys
    CUDA_VISIBLE_DEVICES=3,4
    OPENAI_API_KEY=your_openai_api_key
    HF_TOKEN=your_huggingface_token
    HF_HOME=/path/to/your/huggingface/cache
    ```

## Data

### Inspection

To visibly check whether the RAG questions of a dataset can be answered by the corresponding context, run the following command:

    python3 -m src.inspect.inspect_spans -d <dataset_name> -m <max_samples>

The dataset names can be found in the file itself.  
The file can be used for every dataset that outputs the common `EvalSample` format.
