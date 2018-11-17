# <p style="text-align: center;">Work on AIMA-Python</p>
<p style="text-align: center;">Finish AIMA-python algorithms and add explanatory notebooks

---
Aman Deep Singh
<br>
_Github: ad71_
<br>
_Email: ads6may@gmail.com_

### ABSTRACT
The aim of this project is to finish incomplete implementations of the algorithms in aima-python, add tests for them and add detailed jupyter notebooks explaining the functioning and the usage of the algorithms.

### PROJECT PROPOSAL
1. The aima-python repository currently has 6 incomplete algorithms, one of which is currently being worked on. I propose to complete the implementation of these algorithms and add tests for them. 
2. I will write detailed jupyter notebooks for the algorithms which do not have notebook sections, explaining the intuition behind the algorithms and providing graphing and visualization wherever necessary. These notebooks will include guides and examples for using the algorithms on real-world problems. The notebooks will also address the advantage of certain algorithms over others in different scenarios.
3. Notebook sections for some algorithms are inadequately or incomprehensibly written and require a revisit. I will rewrite some parts to accomodate the ideas in the previous point.
4. I will add tests for the newly added algorithms as well as for the ones that are missing tests. Some tests for the existing algorithms are quite trivial and serve no real purpose. I will add tests for these algorithms too in an attempt to increase coverage.
5. Lower down the priority list, I will write simple GUI applications for some of the classic algorithms as I believe certain concepts are explained best when the student can interact with an application.
6. Some notebooks have a corresponding `apps` notebook. I plan to extend this idea for a few notebooks. For example a `search-apps` notebook will be very helpful as search algorithms can be applied to almost all optimization problems.
7. If time permits, I also plan to remove dependencies on third-party libraries like `networkx`. I will try my best to rewrite these parts using the classes already built into `aima-python`.
8. I have quite a lot of experience with JavaScript as well and I am willing to work on `aima-exercises` too, if required. I have included links to a few of my JavaScript projects in this document.

In case of unforeseen problems, I might be unable to work on all these points, but I will try my best to complete the highest priority tasks. I do not have other commitments this summer and I am willing to work 45-50 hours a week on this project. I do not yet have a plan on how to start working on this project but I have decent knowledge about almost all the concepts in the book and am willing to work however my mentors see fit. Given a choice, I would probably start with the `logic`,  or `mdp` modules.

### MY WORK ON THE REPOSITORY
The following is a summary of my contributions to the project.
<br>
1.	Genetic Algorithm section in `search.ipynb` [#702](https://github.com/aimacode/aima-python/pull/702)
<br>
2.	Value Iteration explanation in `mdp.ipynb` [#736](https://github.com/aimacode/aima-python/pull/736)
<br>
3.	Policy Iteration explanation and various enhancements to `mdp.ipynb` [#743](https://github.com/aimacode/aima-python/pull/743)
<br>
4.	Created `mdp_apps.ipynb` for applications of MDPs. [#743](https://github.com/aimacode/aima-python/pull/743) and [#782](https://github.com/aimacode/aima-python/pull/782)
<br>
5.	Hill-Climbing section in `search.ipynb` [#787](https://github.com/aimacode/aima-python/pull/787)
<br>
6.	TT-Entails and TT-Check-All section in `logic.ipynb` [#793](https://github.com/aimacode/aima-python/pull/793)
<br>
7.	To-CNF explanation in `logic.ipynb` [#802](https://github.com/aimacode/aima-python/pull/802)
<br>
8.	PL-FC-Entails section in `logic.ipynb` [#818](https://github.com/aimacode/aima-python/pull/818)
<br>
9.	DPLL section in `logic.ipynb` [#823](https://github.com/aimacode/aima-python/pull/823)
<br>
10. WalkSAT section in `logic.ipynb` [#823](https://github.com/aimacode/aima-python/pull/823)
<br>
11. Min-Conflicts section in `logic.ipynb` [#841](https://github.com/aimacode/aima-python/pull/841)
<br>
12. Simulated-Annealing section in `search.ipynb` [#866](https://github.com/aimacode/aima-python/pull/866)
<br>
13. Rewrote `EightPuzzle` and `NQueensProblem` sections in `search.ipynb` [#848](https://github.com/aimacode/aima-python/pull/848)
<br>
14. Refactored `EightPuzzle` class to work with search algorithms [#807](https://github.com/aimacode/aima-python/pull/807)
<br>
15. Refactored `NQueensProblem` class to work with search algorithms [#848](https://github.com/aimacode/aima-python/pull/848)
<br>
16. Added tests for `mdp.py` [#722](https://github.com/aimacode/aima-python/pull/722)
<br>
17. Added tests for `EightPuzzle` and `astar_search` [#807](https://github.com/aimacode/aima-python/pull/807)
<br>
18. Added tests for `pl_fc_entails` [#818](https://github.com/aimacode/aima-python/pull/818)
<br>
19. Added tests for `NQueensCSP` and `min_conflicts` [#841](https://github.com/aimacode/aima-python/pull/841)
<br>
20. Added tests for `NQueensProblem`, `breadth_first_tree_search`, `depth_first_tree_search`, `astar_search` and `uniform_cost_search` [#848](https://github.com/aimacode/aima-python/pull/848)
<br>
21. Added tests for `unify`, `tt_entails`, `pl_resolution`, `WalkSAT` and `dpll_satisfiable` [#854](https://github.com/aimacode/aima-python/pull/854)
<br>
22. Changed plotting function for `NQueensCSP` [#847](https://github.com/aimacode/aima-python/pull/847)
<br>
23. `recombine_uniform` function in `search.py` [#704](https://github.com/aimacode/aima-python/pull/704)
<br>
24. Genetic Algorithm GUI example `gui/genetic_algorithm.py` [#702](https://github.com/aimacode/aima-python/pull/702)
<br>
25. Grid-MDP Editor GUI `gui/grid_mdp.py` [#719](https://github.com/aimacode/aima-python/pull/719) and [#734](https://github.com/aimacode/aima-python/pull/734)
<br>
26. Eight-Puzzle GUI `gui/eight_puzzle.py` [#861](https://github.com/aimacode/aima-python/pull/861)
<br>
27. TSP algorithm selection menu (GUI) `gui/tsp.py` [#706](https://github.com/aimacode/aima-python/pull/706)
<br>
28. Minor bugfixes [#715](https://github.com/aimacode/aima-python/pull/715), [#771](https://github.com/aimacode/aima-python/pull/771), [#790](https://github.com/aimacode/aima-python/pull/790), [#794](https://github.com/aimacode/aima-python/pull/794), [#801](https://github.com/aimacode/aima-python/pull/801), [#832](https://github.com/aimacode/aima-python/pull/832), [#864](https://github.com/aimacode/aima-python/pull/864) and [#867](https://github.com/aimacode/aima-python/pull/867)
<br>
29. Solved issues: [#696](https://github.com/aimacode/aima-python/issues/696), [#709](https://github.com/aimacode/aima-python/issues/709), [#710](https://github.com/aimacode/aima-python/issues/710), [#711](https://github.com/aimacode/aima-python/issues/711), [#713](https://github.com/aimacode/aima-python/issues/713), [#766](https://github.com/aimacode/aima-python/issues/766), [#785](https://github.com/aimacode/aima-python/issues/785), [#792](https://github.com/aimacode/aima-python/issues/792), [#824](https://github.com/aimacode/aima-python/issues/824) and [#826](https://github.com/aimacode/aima-python/issues/826)
<br>
30. Helped users solve issues: [#786](https://github.com/aimacode/aima-python/issues/786), [#800](https://github.com/aimacode/aima-python/issues/800), [#819](https://github.com/aimacode/aima-python/issues/819), [#824](https://github.com/aimacode/aima-python/issues/824), [#826](https://github.com/aimacode/aima-python/issues/826), [#865](https://github.com/aimacode/aima-python/issues/865)
<br>
31. Gave feedback and suggestions to users: [#705](https://github.com/aimacode/aima-python/issues/705), [#735](https://github.com/aimacode/aima-python/issues/735), [#814](https://github.com/aimacode/aima-python/issues/814), [#830](https://github.com/aimacode/aima-python/issues/830), [#856](https://github.com/aimacode/aima-python/issues/856), [#858](https://github.com/aimacode/aima-python/issues/858), [#860](https://github.com/aimacode/aima-python/pull/860), [#855](https://github.com/aimacode/aima-python/pull/855), [#846](https://github.com/aimacode/aima-python/pull/846), [#845](https://github.com/aimacode/aima-python/pull/845), [#842](https://github.com/aimacode/aima-python/pull/842), [#837](https://github.com/aimacode/aima-python/pull/837), [#835](https://github.com/aimacode/aima-python/pull/835), [#827](https://github.com/aimacode/aima-python/pull/827), [#825](https://github.com/aimacode/aima-python/pull/825), [#815](https://github.com/aimacode/aima-python/pull/815), [#810](https://github.com/aimacode/aima-python/pull/810), [#808](https://github.com/aimacode/aima-python/pull/808), [#791](https://github.com/aimacode/aima-python/pull/791), [#781](https://github.com/aimacode/aima-python/pull/781), [#776](https://github.com/aimacode/aima-python/pull/776), [#742](https://github.com/aimacode/aima-python/pull/742) and [#737](https://github.com/aimacode/aima-python/pull/737)
<br>

### PAST PROJECTS
1. __CNNs learn to drive__: An ensemble of convolutional neural networks were trained on an image dataset to drive a car on a thin track. The [environment](https://github.com/ad71/Unity-Projects-2/tree/master/Car%20AI%203) was built in Unity-C#. The dataset was generated from a screencast of the game being played by actual humans. The dataset was appropriately preprocessed and fed into an ensemble of convolutional neural networks which was then trained to control a car (using virtual keypresses) on the track using the real time screencast as test data. Source code links: [environment](https://github.com/ad71/Unity-Projects-2/tree/master/Car%20AI%203), [project](https://github.com/ad71/Practical-ML/tree/master/Python%20Plays).

2. __Island project__: This project was done as part of a Hackathon last year. It consisted of an environment with a track on an island. The track was marked with waypoints. Cars with random physical properties were run on it and optimized using modified genetic algorithms in an attempt to find the values that minimize the track-time. There were 18 variables to optimize over, including max steer-angle, max braking torque, throttle delay, etc. Several generations were run and the data was fed into a neural network and trained to predict the track time given the values of the 18  variables. A data visualization tool was also built. A lot of interesting results were found. Repository links for the [environment](https://github.com/ad71/Unity-Projects-2/tree/master/Car%20AI%203) and the [network model](https://github.com/ad71/Practical-ML/tree/master/GAmine).

3. __BoxCar2D__: This is a computation intelligence car evolution project, a replica of [BoxCar2D](http://www.boxcar2d.com/). A 'car' is generated by drawing a polygon using eight randomly selected points as vertices. Each vertex has a probability of spawning a wheel of random size. A population of 20 'cars' are run on a track and their characteristics are optimized using a genetic algorithm to get the optimum model. The environment was completely built on Processing, a Java-based environment and the Box2D physics engine was used. [Directory](https://github.com/ad71/Genetic-Algorithms/tree/master/boxCar2D)

4. __Genetic Algorithms__: I have numerous smaller projects based on Genetic Algorithms, the latest one involving NEAT, where we try to find the best neural network architecture for a particular problem using genetic algorithms. Details are in the README  of [this repository](https://github.com/ad71/Genetic-Algorithms)

5. __Deep Learning__: Deep Learning interests me the most. Following are the topics and repository links of some of my work in this field.
<br>
1	Analysis of linear models using `scikit-learn`. [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/sklearn_linear_models.ipynb) and [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/Intro%20to%20DL/linear_regression_sklearn.ipynb)
<br>
2a	MNIST dataset image classification using CNNs. [program](https://github.com/ad71/Practical-ML/blob/master/tf_convnet4.py)
<br>
2b	MNIST dataset image classification using RNNs. [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/rnn_mnist.ipynb)
<br>
2c	MNIST dataset analysis using the `keras` api. [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/keras_api.ipynb)
<br>
2d	Effect of Adversarial Noise on the MNIST dataset. [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/adversarial_noise_mnist.ipynb)
<br>
3a	An AI to balance a pole on a cart (the `CartPole-v0` environment of OpenAI-gym) using densely connected neural networks. [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/tensorflow_openai.ipynb)
<br>
3b	An AI for the `CartPole-v0` environment using vanilla policy-gradients. [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/Reinforcement%20Learning/vanilla_policy_gradient_agent.ipynb) and [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/Reinforcement%20Learning/cartpole-v0_policy_gradients.ipynb)
<br>
4	Statistical Learning (statistical inference using machine learning) using `scikit-learn`. [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/sklearn_statistical_learning.ipynb)
<br>
5	Data analysis on the Iris dataset. [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/iris.ipynb)
<br>
6	Clustering and compression analysis using `scikit-learn`. [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/sklearn_clustering.ipynb)
<br>
7	KMeans clustering analysis. [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/KMeans.ipynb)
<br>
9	Data analysis on the titanic-dataset using clustering methods. [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/mean_shift_titanic.ipynb), [algorithm](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/mean_shift.py)
<br>
10	Support Vector Machines in `numpy`. [program](https://github.com/ad71/Practical-ML/blob/master/svm-algo.py)
<br>
11	Effect of Adversarial Noise on the `Inception-v3` model. [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/adversarial_noise_inception_v3.ipynb)
<br>
12	Principal component analysis. [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/HR_dataset_PCA.ipynb)
<br>
13	Python implementation of Geoffrey Hinton's [Capsule Network](https://arxiv.org/pdf/1710.09829.pdf) concept using `tensorflow`. [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/CapsNet_tf.ipynb)
<br>
14	Sentiment Analysis on IMDb movies dataset. [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/Intro%20to%20DL/sentiment_analysis_challenge.ipynb)
<br>
15	Music generation from MIDI sequences using RNN (LSTM) networks. [directory](https://github.com/ad71/Data-Science/tree/master/Machine%20Learning/Intro%20to%20DL/Music), [model](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/Intro%20to%20DL/Music/lstm.py)
<br>
16	Transfer Learning with `tensorflow` on the CIFAR-10 dataset using the `inception-v3` model. [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/transfer_learning_inceptionv3-cifar10.ipynb) and another related [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/transfer_learning_video_data.ipynb).
<br>
17	Style transfer on images with `keras` on `tensorflow` using the VGG16 model. [notebook](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/Intro%20to%20DL/style_transfer_keras.ipynb)
<br>
18	AI for the [Halite-2](https://halite.io/) competition. [directory](https://github.com/ad71/Data-Science/tree/master/Machine%20Learning/Halite2) and [directory](https://github.com/ad71/Data-Science/tree/master/Machine%20Learning/Halite2ML).


6. __Game development__: I spent a few months learning to make games on the Unity engine using C#. Simple examples of my work are included in [this repository](https://github.com/ad71/Unity-Projects). Unity was also used to create the [environment](https://github.com/ad71/Unity-Projects-2/tree/master/Car%20AI%20GA) for the "Island project" and the [track](https://github.com/ad71/Unity-Projects-2/tree/master/Car%20AI%203) for the "CNNs learn to drive" project. The source for these environments and a few other simpler games are included in [this repository](https://github.com/ad71/Unity-Projects-2). I also have experience with Python's game engine `pygame` and the physics engine `pymunk`. [Directory](https://github.com/ad71/Data-Science/tree/master/Data%20Visualization/Pygame).

7. __Natural  Language Understanding__: This was a collaborative NLU project (written in C++) based on a [problem statement](https://github.com/sv-cheats1/nlu/blob/master/V2-final/Problem2_final.docx) in the latest [Hackathon](http://hackfest.in/) that I participated in, where we were expected to classify input sentences into their corresponding 'Grammars' and to find the boundaries for placeholders and open-phrases. Given an input string, we were supposed to find out what the user wants and call the respective function with the correct arguments as part of a larger NLU project. Implementational details and the source code are in [this repository](https://github.com/sv-cheats1/nlu).

8. __Diabetic Retinopathy Detection__: Diabetic retinopathy is the  leading cause of blindness in the working-age population. This is an ongoing collaborative project aimed at detecting the presence and severity of this disease using retina scans. This project is incomplete and still in the preliminary phase at the time of writing. [Repository](https://github.com/sv-cheats1/Diabetic-Retinopathy-Detection)

9. __Web development__: This project was aimed at developing a [website](https://jaspalsinghart.com/) for a contemporary Indian artist. [Repository](https://github.com/ad71/Official-Web-Portal).

10. __JavaScript applets__: I sometimes make simple JavaScript applets in my leisure time, some of which are deployed as webapps. [This repository](https://github.com/ad71/General-Programming-P5) contains more details. 

11. __Processing sketches__: I sometimes make simple Processing (a Java-based environment) applications. [This repository](https://github.com/ad71/General-Programming-P3) and [this repository](https://github.com/ad71/Processing-Sketchbook) contain some examples.

12. __Android Application Development__: This was a college-management application developed by me and another contributor during the annual Hackathon in my college last year. Though the app hasn't been deployed, it is functionally complete. [Repository for android app](https://github.com/ad71/Struo-Reloaded), [Repository for web app](https://github.com/ad71/Struo-web).

### ABOUT ME
I am an undergraduate student pursuing Bachelor of Technology in Electronics and Communication at the Indian Institute of Technology (ISM), Dhanbad. My main areas of interest are Artificial Intelligence, Deep Learning, and Calculus. I am fluent in Python, C++ and JavaScript and spend all my free time exploring the different domains of Artificial Intelligence. I have also completed several MOOCs on Machine Learning and Deep Learning. The certificates for the same are linked [here](https://github.com/ad71/Data-Science/blob/master/Machine%20Learning/Coursera.md). 
<br>
It's been quite some time since I am contributing to this project and in my opinion, I have made some sizeable contributions to it. I have learnt a lot from the book, the codebase and my fellow contributors. It would be a great opportunity for me if I am selected to continue working on this project as a part of GSoC this year. I plan to be a long-time contributor and will support the organization and new contributors even after GSoC and when AIMA-4e is eventually released. I would love to help this project under a mentor.

### CONTACT DETAILS
Full name: Aman Deep Singh
<br>
Github: ad71
<br>
E-mail: ads6may@gmail.com, amanverydeep@gmail.com
<br>