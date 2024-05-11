# Working with streamlit.

We try to run the streamlit app on command prompt instead of google colab.

Firstly,


```bash
# Create a new environment
conda create --name streamlit python=3.8
conda activate streamlit
```

## Install the required packages
```bash
pip install streamlit
pip install opencv-python-headless
pip install numpy
pip install torch torchvision
```
Then,

1. Load the torch script models or openvino models in to your app and modify the already existing path.
2. Run the app by `streamlit run app.py`.



