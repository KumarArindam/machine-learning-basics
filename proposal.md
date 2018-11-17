# <p style="text-align: center;">Work on AIMA-Python</p>
<p style="text-align: center;">Finish AIMA-python algorithms and add explanatory notebooks

---
Aman Deep Singh
_Github: ad71_
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
8. I have quite a lot of experience with JavaScript as well and I am willing to work on `aima-exercises` too, if required. I have included links to a few JavaScript projects in this document.

In case of unforeseen problems, I might be unable to work on all these points, but I will try my best to complete the tasks with the highest priority. I do not have other commitments this summer and I am willing to work 45-50  hours a week on this project. I do not yet have a plan on how to start working on this project as I have a decent knowledge about almost all the concepts in the book and am willing to work however my mentors see fit. Given a choice, I would probably start with the `logic`,  or `mdp` modules.

### MY WORK ON THE REPOSITORY
The following is a summary of my contributions to the project.
1.	Genetic Algorithm section in `search.ipynb` [#702](https://github.com/aimacode/aima-python/pull/702)
2.	Value Iteration explanation in `mdp.ipynb` [#736](https://github.com/aimacode/aima-python/pull/736)
3.	Policy Iteration explanation and various enhancements to `mdp.ipynb` [#743](https://github.com/aimacode/aima-python/pull/743)
4.	Created `mdp_apps.ipynb` for applications of MDPs. [#743](https://github.com/aimacode/aima-python/pull/743) and [#782](https://github.com/aimacode/aima-python/pull/782)
5.	Hill-Climbing section in `search.ipynb` [#787](https://github.com/aimacode/aima-python/pull/787)
6.	TT-Entails and TT-Check-All section in `logic.ipynb` [#793](https://github.com/aimacode/aima-python/pull/793)
7.	To-CNF explanation in `logic.ipynb` [#802](https://github.com/aimacode/aima-python/pull/802)
8.	PL-FC-Entails section in `logic.ipynb` [#818](https://github.com/aimacode/aima-python/pull/818)
9.	DPLL section in `logic.ipynb` [#823](https://github.com/aimacode/aima-python/pull/823)
10. WalkSAT section in `logic.ipynb` [#823](https://github.com/aimacode/aima-python/pull/823)
11. Min-Conflicts section in `logic.ipynb` [#841](https://github.com/aimacode/aima-python/pull/841)
12. Simulated-Annealing section in `search.ipynb` [#866](https://github.com/aimacode/aima-python/pull/866)
13. Rewrote `EightPuzzle` and `NQueensProblem` sections in `search.ipynb` [#848](https://github.com/aimacode/aima-python/pull/848)
14. Refactored `EightPuzzle` class to work with search algorithms [#807](https://github.com/aimacode/aima-python/pull/807)
15. Refactored `NQueensProblem` class to work with search algorithms [#848](https://github.com/aimacode/aima-python/pull/848)
16. Added tests for `mdp.py` [#722](https://github.com/aimacode/aima-python/pull/722)
17. Added tests for `EightPuzzle` and `astar_search` [#807](https://github.com/aimacode/aima-python/pull/807)
18. Added tests for `pl_fc_entails` [#818](https://github.com/aimacode/aima-python/pull/818)
19. Added tests for `NQueensCSP` and `min_conflicts` [#841](https://github.com/aimacode/aima-python/pull/841)
20. Added tests for `NQueensProblem`, `breadth_first_tree_search`, `depth_first_tree_search`, `astar_search` and `uniform_cost_search` [#848](https://github.com/aimacode/aima-python/pull/848)
21. Added tests for `unify`, `tt_entails`, `pl_resolution`, `WalkSAT` and `dpll_satisfiable` [#854](https://github.com/aimacode/aima-python/pull/854)
22. Changed plotting function for `NQueensCSP` [#847](https://github.com/aimacode/aima-python/pull/847)
23. `recombine_uniform` function in `search.py` [#704](https://github.com/aimacode/aima-python/pull/704)
24. Genetic Algorithm GUI example `gui/genetic_algorithm.py` [#702](https://github.com/aimacode/aima-python/pull/702)
25. Grid-MDP Editor GUI `gui/grid_mdp.py` [#719](https://github.com/aimacode/aima-python/pull/719) and [#734](https://github.com/aimacode/aima-python/pull/734)
26. Eight-Puzzle GUI `gui/eight_puzzle.py` [#861](https://github.com/aimacode/aima-python/pull/861)
27. TSP algorithm selection menu (GUI) `gui/tsp.py` [#706](https://github.com/aimacode/aima-python/pull/706)
28. Minor bugfixes [#715](https://github.com/aimacode/aima-python/pull/715), [#771](https://github.com/aimacode/aima-python/pull/771), [#790](https://github.com/aimacode/aima-python/pull/790), [#794](https://github.com/aimacode/aima-python/pull/794), [#801](https://github.com/aimacode/aima-python/pull/801), [#832](https://github.com/aimacode/aima-python/pull/832), [#864](https://github.com/aimacode/aima-python/pull/864) and [#867](https://github.com/aimacode/aima-python/pull/867)
29. Solved issues: [#696](https://github.com/aimacode/aima-python/issues/696), [#709](https://github.com/aimacode/aima-python/issues/709), [#710](https://github.com/aimacode/aima-python/issues/710), [#711](https://github.com/aimacode/aima-python/issues/711), [#713](https://github.com/aimacode/aima-python/issues/713), [#766](https://github.com/aimacode/aima-python/issues/766), [#785](https://github.com/aimacode/aima-python/issues/785), [#792](https://github.com/aimacode/aima-python/issues/792), [#824](https://github.com/aimacode/aima-python/issues/824) and [#826](https://github.com/aimacode/aima-python/issues/826)
30. Helped users solve issues: [#786](https://github.com/aimacode/aima-python/issues/786), [#800](https://github.com/aimacode/aima-python/issues/800), [#819](https://github.com/aimacode/aima-python/issues/819), [#824](https://github.com/aimacode/aima-python/issues/824), [#826](https://github.com/aimacode/aima-python/issues/826), [#865](https://github.com/aimacode/aima-python/issues/865)
31. Gave feedback and suggestions to users: [#705](https://github.com/aimacode/aima-python/issues/705), [#735](https://github.com/aimacode/aima-python/issues/735), [#814](https://github.com/aimacode/aima-python/issues/814), [#830](https://github.com/aimacode/aima-python/issues/830), [#856](https://github.com/aimacode/aima-python/issues/856), [#858](https://github.com/aimacode/aima-python/issues/858), [#860](https://github.com/aimacode/aima-python/pull/860), [#855](https://github.com/aimacode/aima-python/pull/855), [#846](https://github.com/aimacode/aima-python/pull/846), [#845](https://github.com/aimacode/aima-python/pull/845), [#842](https://github.com/aimacode/aima-python/pull/842), [#837](https://github.com/aimacode/aima-python/pull/837), [#835](https://github.com/aimacode/aima-python/pull/835), [#827](https://github.com/aimacode/aima-python/pull/827), [#825](https://github.com/aimacode/aima-python/pull/825), [#815](https://github.com/aimacode/aima-python/pull/815), [#810](https://github.com/aimacode/aima-python/pull/810), [#808](https://github.com/aimacode/aima-python/pull/808), [#791](https://github.com/aimacode/aima-python/pull/791), [#781](https://github.com/aimacode/aima-python/pull/781), [#776](https://github.com/aimacode/aima-python/pull/776), [#742](https://github.com/aimacode/aima-python/pull/742) and [#737](https://github.com/aimacode/aima-python/pull/737)

### PAST PROJECTS
1. <placeholder car ga project>
2. <placeholder car cnn and ai project>__CNNs learn to drive__: An ensemble of convolutional neural networks were trained on an image dataset to drive a car on a thin track. The [environment](https://github.com/ad71/Unity-Projects-2/tree/master/Car%20AI%203) was built in Unity-C#. The dataset was generated from a screencast of the game being played by actual humans. The dataset was appropriately preprocessed and fed into an ensemble of convolutional neural networks which was then trained to control a car (using virtual keypresses) on the track using the real time screencast as test data. Source code links: [environment](https://github.com/ad71/Unity-Projects-2/tree/master/Car%20AI%203), [project](https://github.com/ad71/Practical-ML/tree/master/Python%20Plays).
3. <placeholder boxcar2d>
4. <palceholder genetic algorithms>
5. <placeholder unity games> __Game development__: I spent a few months learning to make games on the Unity engine using C#. Simple examples of my work are included in [this repository]. Unity was also used to create the [environment](https://github.com/ad71/Unity-Projects-2/tree/master/Car%20AI%20GA) for the Island project and the [track](https://github.com/ad71/Unity-Projects-2/tree/master/Car%20AI%203) for the "CNNs learn to drive" project. The source for these environments and a few other simpler games are included in [this repository](https://github.com/ad71/Unity-Projects-2).
6. <placeholder transfer learning>
7. <placeholder deep learning projects>
8. <placeholder nlu project>
9. <placeholder website> __Web development__: I made a [website](https://jaspalsinghart.com/) for contemporary Indian artist Jaspal Singh. [Repository](). <Include repository link>
10. <placeholder JavaScript sketches>__JavaScript applets__: I sometimes make simple JavaScript applets in my leisure time some of which are deployed as webapps. [This repository]() contains more details.
11. <placeholder Processing sketches>
12. <placheolder diabetic retinopathy detection>__Diabetic Retinopathy Detection__: Diabetic retinopathy is the  leading cause of blindness in the working-age population. This is an ongoing collaborative project aimed at detecting the presence and severity of this disease using retina scans. This project is still in the data-preprocessing phase at the time of writing. [Repository]() <Include repository link>
13. <placeholder struo>

### ABOUT ME
I am an undergraduate student pursuing Bachelor of Technology in Electronics and Communication at the Indian Institute of Technology (ISM), Dhanbad. My main areas of interest are Artificial Intelligence, Deep Learning, Data Science and Calculus. I am fluent in Python and C++ and spend all my free time exploring the different domains of Artificial Intelligence. I would love to help this project under a mentor.

