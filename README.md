# Flower-Classifier

Welcome to my personal flowery hell that I brought to myself

This code is a combination of a youtube video which Ill link here:
https://www.youtube.com/watch?v=il8dMDlXrIE
and my best friend chatgpt helping me add features.

Here is the flower dataset:
https://drive.google.com/drive/folders/19S90PI1hD7vkQ1rSUiC8PUbUNPM7BhTN?usp=sharing

Now to the actual code

just download the Flowers folder it (as of June 15 10:59pm PH time)
has 6 flowers that are found in the Philippines at the moment.

Run the code, and select an image that you want classified.

# How to Run

run main.py first if model.h5 isn't there or if you made adjustments to the dataset or the training process

then after run:
streamlit run app.py

Happy Classifying!

# My notes

TBH you can also just repurpose the code not just for flowers, as long as you have the clarity to
change the following:

input_dir = input_dir = './Flowers'

and

categories = ['gumamela', 'sunflower', 'tulip']
to whatever you name your folders.

It should work without a hitch.

## Notes from the future

Yes I came back to this project because I didn't expect it to not work at all on my laptop
so severe changes had to be made to
get it to work on my laptop including:
- Using env becasue python incompatibility
- Transition to Streamlit for selecting images (mac issue)