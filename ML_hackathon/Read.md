# ML Hackathon - Hackman: Hybrid Model using HMM and Reinforcement Learning (Q-Learning)

## Overview
This project presents an **AI-powered Hangman game** that learns to guess words intelligently by combining **Hidden Markov Models (HMM)** with **Reinforcement Learning (Q-Learning)**. The system predicts likely letters using statistical language modeling (HMM) and then optimizes its decision-making strategy through trial and error (RL). A **Streamlit web app** is also included for interactive gameplay where users can play against the trained AI.


## Team
Developed by:  

**Diya D Bhat**  
**Dhanya Prabhu**  
**Delisha Riyona Dsouza**
**Eshwar R A**  


## Project Description
This work explores a **hybrid AI approach** that merges probabilistic reasoning (HMM) and sequential decision-making (Reinforcement Learning) to tackle the Hangman word-guessing challenge.

- The **Hidden Markov Model (HMM)** learns letter patterns and transition probabilities from a large English word corpus. It estimates which letters are most probable given the current word pattern.
- The **Q-Learning agent** uses these HMM-generated probabilities as part of its feature set to learn how to guess letters optimally. Over multiple episodes, it learns through rewards and penalties to balance exploration (trying new letters) and exploitation (choosing high-confidence guesses).

This hybrid design ensures the model benefits from both **linguistic understanding** and **experience-based learning**. The final Streamlit interface allows users to **interactively test or challenge the AI** and visualize its decision-making process.


## Dataset
The dataset used for training the HMM is a **text-based corpus of English words**.  

Two files are used:
- **corpus.txt** → Training data for the HMM  
- **test.txt** → Evaluation set for model testing

Both files are text-based lists of words, one per line.


## Running the Project

### Step 1: Obtaining Access to the Codes

#### Download the Repository as ZIP
1. In this repository, click the green **Code** button and select **Download ZIP**.  
2. Extract the ZIP file locally.  
3. Navigate to the project folder and open it in your preferred IDE or terminal.
4. to train use python hangman_ai_full.py train
5. evaulate it using python3 hangman_ai_full.py evaluate
6. for the ui run :streamlit run app.py     

### Step 2: Running the files
1. Open the ipynb file provided and run it cell by cell.
2. Once training is complete, pkl files are generated that can be used for inference or integration with the frontend.
> **Note:**  
> A **frontend interface using Streamlit** (app.py) was developed for interactive gameplay and model visualization.
> TO access the frontend , download the 2 files , app.py and hungman_ai_full.py 
> train and evaluate using the above given commands , and then use streamlit run app.py to run the ui
