# CSCI435 Project

## Installation

To run the project, install the required dependencies using:

```bash
pip install -r requirements.txt
```

**Note:** One of the requirements is `face_recognition`, which requires `dlib` and `cmake`. If you face issues installing it, you can refer to the [face_recognition installation guide](https://github.com/ageitgey/face_recognition#installation).

## Running the Project

### Jupyter Notebook

The main Jupyter notebook is `CSCI435_Project.ipynb`. You can run the cells in the notebook to execute the code.

**Note:** The notebook uses `cv2.imshow` to display images, so make sure you have a compatible environment for displaying images.

### Streamlit App

To run the Streamlit app, use the following command:

```bash
streamlit run app.py
```
