import streamlit as st
import random
import numpy as np
import pickle
from hangman_ai_full import HangmanHMM, HangmanQLearningAgent, HangmanEnvironment

# ----------------------------------------------------------
# Load trained models
# ----------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        hmm = HangmanHMM()
        hmm.load("hangman_hmm.pkl")
        agent = HangmanQLearningAgent(hmm)
        agent.load("hangman_agent.pkl")
        st.success("Models loaded successfully!")
        return hmm, agent
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

hmm, agent = load_models()

# ----------------------------------------------------------
# Initialize Streamlit UI
# ----------------------------------------------------------
st.set_page_config(page_title="Hangman AI", layout="centered")
st.title("Hangman AI — HMM + Reinforcement Learning Demo")
st.caption("A hybrid AI that learns to play Hangman using Hidden Markov Models + Q-Learning")

# ----------------------------------------------------------
# Choose or reset a game
# ----------------------------------------------------------
test_words = ["planet", "orange", "stream", "python", "school", "system", "banana"]
user_word = st.text_input("Enter a word to test (or leave blank for random):")

if st.button("Start / Reset Game"):
    chosen_word = user_word.strip().lower() if user_word.strip() else random.choice(test_words)
    st.session_state.game = HangmanEnvironment(chosen_word)
    st.session_state.state = st.session_state.game.reset()
    st.session_state.last_message = ""
    st.session_state.agent_turn = False

if "game" not in st.session_state:
    st.session_state.game = HangmanEnvironment(random.choice(test_words))
    st.session_state.state = st.session_state.game.reset()
    st.session_state.last_message = ""
    st.session_state.agent_turn = False

game = st.session_state.game
state = st.session_state.state

# ----------------------------------------------------------
# Display current game state
# ----------------------------------------------------------
st.markdown(f"## Word: {' '.join([c for c in state['masked_word']])}")
st.markdown(f"*Guessed letters:* {', '.join(sorted(state['guessed_letters'])) or 'None'}")
st.markdown(f"*Wrong guesses:* {state['wrong_guesses']} / {game.max_wrong_guesses}")

if st.session_state.last_message:
    st.info(st.session_state.last_message)

# ----------------------------------------------------------
# Player + Agent actions (disabled after game over)
# ----------------------------------------------------------
col1, col2 = st.columns([2, 1])

game_over = game.game_over or state["wrong_guesses"] >= game.max_wrong_guesses

with col1:
    if "current_guess" not in st.session_state:
        st.session_state.current_guess = ""

    guess = st.text_input(
        "Enter your guess (a–z):",
        value=st.session_state.current_guess,
        max_chars=1,
        key="guess_input",
    )

    # Disable the button when game is over
    submit_disabled = game_over

    if st.button("Submit Guess", key="submit_guess", disabled=submit_disabled):
        if not guess or not guess.isalpha() or len(guess) != 1:
            st.warning("Please enter a valid letter.")
        else:
            next_state, reward, done = game.step(guess.lower())
            st.session_state.state = next_state
            st.session_state.last_message = (
                "Correct guess!" if reward > 0 else "Wrong guess!"
            )
            st.session_state.current_guess = ""
            st.rerun()

with col2:
    ai_disabled = game_over
    if st.button("Let Model Guess", key="ai_guess", disabled=ai_disabled):
        if hmm is None or agent is None:
            st.warning("Models not loaded. Train or load models first.")
        else:
            available = game.get_available_actions()
            if not available:
                st.warning("No more available guesses!")
            else:
                action = agent.choose_action(state, available, training=False)
                next_state, reward, done = game.step(action)
                st.session_state.state = next_state
                st.session_state.last_message = (
                    f"Model guessed *{action.upper()}* → {'Correct' if reward > 0 else 'Wrong'}"
                )
                st.rerun()

# ----------------------------------------------------------
# End of game check
# ----------------------------------------------------------
if game.game_over or state["wrong_guesses"] >= game.max_wrong_guesses:
    if game.won:
        st.balloons()
        st.success(f"The word was *{game.word.upper()}* — You Win!")
    else:
        st.error(f"Game Over! The word was *{game.word.upper()}*.")

    if st.button("Play Again"):
        st.session_state.game = HangmanEnvironment(random.choice(test_words))
        st.session_state.state = st.session_state.game.reset()
        st.session_state.last_message = ""
        st.rerun()