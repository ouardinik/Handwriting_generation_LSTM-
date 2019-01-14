A couple of remarks

- I decided to only focus on the exercise number 1, I am very new to RNN, so I had to spend some time reading articles and courses to fully understand the theory, which left only a short time for the actual implementation.
- to train the model, please run the jupyter notebook called "model_and_training.ipynb".
- to generate strokes unconditionally, I used saved tensorflow sessions. But because of a bug that I couldn't resolve on time (a bug with the placeholders), this function in the dummy.py file is not working and after spending a lot of time on it I decided to run the strokes generation on the notebook as well ( please see the last cell after the training, the code is the same as on dummy.py,but uses the current tf session)
- in the folder utils you will find a file gmm.py where I implemented functions that I  use to compute the loss of the gaussian mixture model. you will also find a python file called batch_generator.py that I will usedto preprocess the data (truncating or padding to fix the length of the input arrays) and generate mini_batches.
-For the architecture I chose to implement the single layer (900 units) LSTM describes in Graves's paper.The actual results are pretty bad, it doesn't look like Alex Graves samples on his paper.The training was very slow (around 8 hrs on CPU) so I could only train the network once, wich left little place for experiments and fine tuning.

I thought that this exercise was very challenging, but I learned a great deal in just a week. Thank you very much :)!
