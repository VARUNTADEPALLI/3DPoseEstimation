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
2. Navigate the app.py file in the repository
3. Run the app by `streamlit run app.py`.

[Here](https://drive.google.com/file/d/1jI6gnrS80kuwL2LxgSqei4d1UeNPXT72/view?usp=sharing) is the demo video for streamlit app.
