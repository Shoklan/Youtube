# Practical Deep Learning for Coders 2020
## Jeremy Howard
## Sylvain Gugger
## Rachel Thomas

1. [Lesson 1: Everyone Can Do World Class](#lession1)
2. [Lesson 2: Getting Our Feet Wet With Ideas and Training](#lession2)
3. [Lesson 3](#lession3)
4. [Lesson 4](#lession4)
5. [Lesson 5](#lession5)
6. [Lesson 6](#lession6)
7. [Lesson 7](#lession7)
8. [Lesson 8](#lession8)


# <a name="lession1">Lesson 1</a>
- This class is intended to be a definitive version for the Course.
- There is a book for the course now.
- The the course follows the book closely; you can also download it for free if you wanted.
- **Don't be an asshole and convert the Notebooks to books.**
![Class Dependencies](images/what-you-dont-need.png)
![Application Domains](images/dl-application-space.png)
- **Deep Learning** comes out of *Neural Networks* which was the work of Warren McCulloch and Walter Pitts in 1943.
- Their work was built onto by **Frank Rosenblatt** who claimed "we are about to witness the birth of such a machine - a machine capable of perceiving, recognizing and identifying its surroundings without any human training or control".
- An MIT professor Marvin Minsky published a book called *Perceptrons* which showed that a single neuron was unable to learn basic mathematical ideas.
- Much happened and in 1986 MIT released a series of books called *Parallel  Distributed Processing*.
- While Jeremy was using them around 1980, some researchers 30 years ago had pointed out that to get good performance you would need more layers.
- To learn, we're going to use:
  1. Play the whole game.
  2. Make the game worth playing.
  3. Work on the hard parts.
- The software stack is:
  1. Fastai on top.
  2. PoyTorch in the middle.
  3. Python at the bottom.
- We will be using **PyTorch** instead of **Tensorflow** since it is faster.
- "PyTorch doesn't have higher level APIs, so we built Fastai."
- You will need a GPU machine to run the examples - well.
- Please use one of the platforms provided instead of your machine since it will be easier.
- **If you're using something that is not free than please shut it down.**
- The forums are very important because that is where all the discussion will take place.
- **I will not be annotating anything about Jupyter Notebooks since I'm accustom to using them.**
![Keyboard Shortcuts](images/keyboard-shortcuts.png)
- First block:
```python
# CLICK ME
from fastai.vision.all import *
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```
- Don't worry about understanding the code yet.
- We created a widgets function which can allow us to update files:
```python

uploader = widgets.FileUpload()
uploader

# Get the image:
img = PILImage.create(uploader.data[0])

# Make a prediction:
img = PILImage.create(uploader.data[0])
is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
```
- Machine learning is like regular programming: a way to get computers to complete a specific task.
- The normal model of coding:
![Model of Coding](images/model-of-programming.png)
- **Arthur Samuel** started working on different ways to get computers to complete tasks.
- He thought that we should feed examples to a computer and let it solve it itself.
![Machine Learning Model of Programming](images/machine-learning-model.png)
- **Inference** is using a trained model to do a task.
  * This is not the traditional definition of *inference*.
- Neural Networks are flexible enough to do any task per the **Universal Approximation Theorem**.
- But to do that, we will need a way to update the weights - which is done via **Stochastic Gradient Descent**.
- The terminology has changed but the model is the same:
![Current Model and Terms](images/terms-and-functions-model.png)
- There are some limitations:
  * A model cannot be created without data.
  * A model can only learn from the patterns it sees in the input Data.
  * The model makes *predictions*, but does not recommend *actions*.
  * It is not enough to have data; you also need labels for the data.
- **Be wary of proxies which represent the values you actually care about**.
- "We spent a lot of time investing in how to allow you to import * without [ combinatorially ] importing everything."
- While we're importing *fastai2*, this version is a pre-release version and *fastai* will be swapped to this latest version.
- There are four defined domains in Fastai:
  1. Vision.
  2. Text.
  3. Tabular.
  4. Collaborative Filtering.
- If you type the function without calling it then it will tell you where it comes from in a Notebook.
- You can call the `doc()` function to get the documentation for a function.
- All of the documentation is fully run-able Jupyter Notebooks.
- When training, you have to tell it:
  1. What data to use?
  2. What architecture to use?
  3. What metrics will be printed out?
- The parameter `valid_pct` will specify how much of the data set should be validation.
- This is to fight **Overfitting**:
![Example of Overfitting](images/example-of-overfit-line.png)
- The library is built so that a similar structure of code can be used for multiple domains.


# <a name="lession2">Lesson 2</a>
- We're going to be looking at Training and Validation more.
- Of note, the `label_func` paramter must return a logical value.
- When you are trying to predict a category, we call that a **Classification Model.**
- Any time that you are trying to predict a number, we call that **Regression**.
- **Regression is not Linear Regression; this is a mistake**.
- After you train for a while, the model predictions will get worse and this is due to **Overfitting**.
- We'll be talking about a learner later but it's a container for the data and the architecture.
- **Resnet** is the kind of architecture; 34 is the number of layers.
- An **Epoch** is what it is called when you look at every single image in the data set once.
- The **Loss** is the not exactly the same thing as your metric,
- For the *loss function*, you need something where you can change by a little bit higher or lower and measure the change.
- The **Metric** is the thing that you care about; the **Loss** is what your model cares about to update parameters.
- **Overfitting** is basically when the model is cheating by identifying patterns in an item vs the category.
- However, sometimes over time the model can be learn even the validation set and sometimes it is a good idea to set aside a third subset: the **Test Set**.
- Be wary of time series data and building validation,test sets on random data.
- Next, we need to learn about the next line of code and **Transfer Learning**:
```python
learn.fine_tune(1)
```
- **Transfer Learning** is using a pretrained model for a different task than what it was originally trained for.
- It turns out that if you use an already trained model and train your model by running more epochs that you end up with a far more accurate model than if you would have done it without.
- Visualization turns out to be important to getting great results.
- From the paper by Matt Zeiler and Rob Fergus, found a way to draw a picture of the first layer.
- What they found is that the first layer founds diagonal lines and gradients - which are simple.
- Layer two takes the features of layer one and combines them into more abstract representations.
![CNN Internal Filters](images/layer-two-internal-cnn.png)
- And, if you keep the earlier layers then you get those identification kernels for free.
- These methods are not just good at recognizing phones but extend beyond that.
- There are all kinds of things which can be transformed into pictures.
![Sound Detection](images/cnn-sound-detect-state-of-the-art.png)
- **Make sure to memorize these terms and their meanings**:
![Terms and Definitions](images/important-terms-and-meanings.png)
![ State of the Art Overview](images/state-of-the-art-overview-image.png)
- The difference between a **Recommendation** ( something similar we don't know about ) and **Prediction** (something we're already aware of) matters in many selling contexts.
- When claiming results, we need to be aware of when a result would fall inside of normal values for parameters.
- We use a **P-Value** to measure how likely it is that our data is random or real:
![P-Value Reasoning](images/flow-of=p-value.png)
![P-Value Distribution](images/p-value-distribution-aka-normal.png)
- P Values are terrible:
![P Values Being Denied by ASA](images/p-values-are-garbo.png)
- Jeremy Developed a different want to approach Data Products called the **Drive Train Approach**.
![Drive Train Approach](images/drive-train-approach-product.png)
- **Autocorrelates** are variables which are are closely related.
- Think about this from a Utility point of view; what actions can I take?:
![Consider Possibilities](images/chart-relate-outcomes-possibilites.png)
- When building models, we try and think about prior beliefs.
- **I am not signing up for Bing Image Search.**
- Something not mentioned is that it is common for categories to be in their own folders - as seen from the file structure in the image:
![Image Paths](images/category-file-structure.png)
- You can verify is something is an image using the function:
```python
fns = get_image_files(path)
failed = verify_images(fns)
failed
```
- You can delete those items that fail using:
```python
failed.map(Path.unlink);
```
- The first thing we need to do is tell Fastai what kind of data we have and how it is structured.
- We're going to be looking at the DataBlock API for data set creation.
![Datablock API](images/data-block-api.png)
```python

bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock),   # What kind of data and labels is this?
    get_items=get_image_files,            # how do we get the data?
    splitter=RandomSplitter(valid_pct=0.2, seed=42), # how do we split the data?
    get_y=parent_label,                   # use the directory the image sits in to label it.
    item_tfms=Resize(128))                # what transforms will be applied?
```
- This is the most common way that images are stored for model creation.
- This all gets put into a **Data Loader** which is something in PyTorch which gets a set of images at a time.
- You can create a **Confusion Matrix** for the categories after training:
```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```
- You can also see what were the offending images using:
![Show Top Losses](images/interp-show-top-losses.png)
```python
interp.plot_top_losses(5, nrows=1)
```
- It will also print images which it is most confident about.
- Getting a model into production, we will export and import it later:
```python
learn.export()

# Pull it back in later:
learn_inf = load_learner(path/'export.pkl')

# predictions:
learn_inf.predict('images/grizzly.jpg')
```


# <a name="lession3">Lesson 3</a>

# <a name="lession4">Lesson 4</a>

# <a name="lession5">Lesson 5</a>

# <a name="lession6">Lesson 6</a>

# <a name="lession7">Lesson 7</a>

# <a name="lession8">Lesson 8</a>

# Research:
- Mark I Perceptron?
- *Perceptrons* by Marvin Minsky?
- Professor David Perkins?
- what is `gv()`?
- Domain specific pretrained models?
- Who is Frank Harrell?
-



# Reference
- [Fastai Online Book](https://github.com/fastai/fastbook)
- [Paper: **Fastai: A Layered API for Deep Learning**](papers/Layered-API-for-Deep-Learning.pdf)
- [Paper: **Artificial Intelligence: A Frontier in Automation**](papers/Artificial-Intelligence-A-Frontier-of-Autmation.pdf)
- [Paper: **Multilayer Feedforward Networks are Universal Approximators**](papers/Multilayer-Feedforward-Networks-are-Universal-Approximators.pdf)
- [Paper: **Visualizing and Understanding Convolutional Networks.pdf**](papers/Visualizing-and-Understanding-Convolutional-Networks.pdf)
- [Model Zoo](https://modelzoo.co/)
