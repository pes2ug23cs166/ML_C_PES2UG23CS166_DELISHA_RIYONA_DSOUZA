# Hangman AI: HMM + Reinforcement Learning Hybrid System
# Complete implementation for the Hangman challenge

import numpy as np
import random
from collections import defaultdict, Counter
import pickle
from typing import List, Set, Tuple, Dict

# ============================================================
# Part 1: Hidden Markov Model
# ============================================================

class HangmanHMM:
    """
    Hidden Markov Model for Hangman letter prediction.
    
    Design:
    - Hidden states: Positions in words
    - Emissions: Letters (a-z)
    - Separate models per word length for accuracy
    """
    
    def __init__(self, max_length=20):
        self.max_length = max_length
        self.emission_probs = {}
        self.transition_probs = {}
        self.initial_probs = {}
        self.letter_freq = Counter()
        
    def train(self, word_list: List[str]):
        """Train HMM on corpus of words"""
        print("Training HMM on corpus...")
        
        # Group words by length
        words_by_length = defaultdict(list)
        for word in word_list:
            word = word.lower().strip()
            if word.isalpha():
                words_by_length[len(word)].append(word)
                for char in word:
                    self.letter_freq[char] += 1
        
        # Train model for each length
        for length, words in words_by_length.items():
            if length > self.max_length or length == 0:
                continue
            
            self.emission_probs[length] = {}
            self.transition_probs[length] = {}
            
            # Count emissions: position -> letter -> count
            emission_counts = {}
            for pos in range(length):
                emission_counts[pos] = Counter()
            
            initial_counts = Counter()
            
            # Count transitions: position -> prev_letter -> next_letter -> count
            transition_counts = {}
            for pos in range(1, length):
                transition_counts[pos] = {}
            
            # Count occurrences
            for word in words:
                for pos, char in enumerate(word):
                    emission_counts[pos][char] += 1
                    
                    if pos == 0:
                        initial_counts[char] += 1
                    else:
                        prev_char = word[pos-1]
                        if prev_char not in transition_counts[pos]:
                            transition_counts[pos][prev_char] = Counter()
                        transition_counts[pos][prev_char][char] += 1
            
            # Convert to probabilities with Laplace smoothing
            for pos in range(length):
                total = sum(emission_counts[pos].values())
                self.emission_probs[length][pos] = {}
                for char in 'abcdefghijklmnopqrstuvwxyz':
                    count = emission_counts[pos].get(char, 0)
                    self.emission_probs[length][pos][char] = (count + 1) / (total + 26)
            
            # Transition probabilities
            for pos in range(1, length):
                self.transition_probs[length][pos] = {}
                for prev_char in 'abcdefghijklmnopqrstuvwxyz':
                    if prev_char in transition_counts[pos]:
                        total = sum(transition_counts[pos][prev_char].values())
                        self.transition_probs[length][pos][prev_char] = {}
                        for char in 'abcdefghijklmnopqrstuvwxyz':
                            count = transition_counts[pos][prev_char].get(char, 0)
                            self.transition_probs[length][pos][prev_char][char] = (count + 1) / (total + 26)
                    else:
                        self.transition_probs[length][pos][prev_char] = {
                            char: 1/26 for char in 'abcdefghijklmnopqrstuvwxyz'
                        }
            
            # Initial probabilities
            total_initial = sum(initial_counts.values())
            self.initial_probs[length] = {}
            for char in 'abcdefghijklmnopqrstuvwxyz':
                count = initial_counts.get(char, 0)
                self.initial_probs[length][char] = (count + 1) / (total_initial + 26)
        
        print(f"✓ HMM trained on {len(word_list)} words")
        print(f"  Lengths covered: {min(words_by_length.keys())} to {max(words_by_length.keys())}")
        
    def predict_letter_probs(self, masked_word: str, guessed_letters: Set[str]) -> Dict[str, float]:
        """Predict probability distribution over remaining letters"""
        length = len(masked_word)
        letter_scores = defaultdict(float)
        
        # Fallback for unseen lengths
        if length not in self.emission_probs:
            total_freq = sum(self.letter_freq.values())
            for char in 'abcdefghijklmnopqrstuvwxyz':
                if char not in guessed_letters:
                    letter_scores[char] = self.letter_freq.get(char, 1) / total_freq
            return dict(letter_scores)
        
        # Accumulate probabilities from each blank position
        for pos, char in enumerate(masked_word):
            if char == '_':
                # Emission probabilities
                for letter in 'abcdefghijklmnopqrstuvwxyz':
                    if letter not in guessed_letters:
                        letter_scores[letter] += self.emission_probs[length][pos].get(letter, 1/26)
                
                # Transition probabilities (if previous char known)
                if pos > 0 and masked_word[pos-1] != '_':
                    prev_char = masked_word[pos-1]
                    if prev_char in self.transition_probs[length][pos]:
                        for letter in 'abcdefghijklmnopqrstuvwxyz':
                            if letter not in guessed_letters:
                                trans_prob = self.transition_probs[length][pos][prev_char].get(letter, 1/26)
                                letter_scores[letter] += trans_prob * 1.5
        
        # Normalize
        total_score = sum(letter_scores.values())
        if total_score > 0:
            letter_scores = {k: v/total_score for k, v in letter_scores.items()}
        else:
            remaining = [c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in guessed_letters]
            letter_scores = {c: 1/len(remaining) for c in remaining}
        
        return dict(letter_scores)
    
    def save(self, filename: str):
        """Save trained HMM to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'emission_probs': self.emission_probs,
                'transition_probs': self.transition_probs,
                'initial_probs': self.initial_probs,
                'letter_freq': self.letter_freq,
                'max_length': self.max_length
            }, f)
        print(f"✓ HMM saved to {filename}")
    
    def load(self, filename: str):
        """Load trained HMM from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.emission_probs = data['emission_probs']
            self.transition_probs = data['transition_probs']
            self.initial_probs = data['initial_probs']
            self.letter_freq = data['letter_freq']
            self.max_length = data['max_length']
        print(f"✓ HMM loaded from {filename}")


# ============================================================
# Part 2: Hangman Environment
# ============================================================

class HangmanEnvironment:
    """Hangman game environment for RL agent"""
    
    def __init__(self, word: str, max_wrong_guesses: int = 6):
        self.word = word.lower()
        self.max_wrong_guesses = max_wrong_guesses
        self.reset()
        
    def reset(self):
        """Reset game state"""
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.repeated_guesses = 0
        self.game_over = False
        self.won = False
        return self.get_state()
    
    def get_state(self) -> Dict:
        """Get current game state"""
        masked_word = ''.join([c if c in self.guessed_letters else '_' for c in self.word])
        return {
            'masked_word': masked_word,
            'guessed_letters': self.guessed_letters.copy(),
            'wrong_guesses': self.wrong_guesses,
            'lives_remaining': self.max_wrong_guesses - self.wrong_guesses,
            'game_over': self.game_over,
            'won': self.won,
            'word_length': len(self.word)
        }
    
    def step(self, letter: str) -> Tuple[Dict, float, bool]:
        """Take action (guess letter)"""
        letter = letter.lower()
        
        # Repeated guess penalty
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            return self.get_state(), -2, self.game_over
        
        self.guessed_letters.add(letter)
        
        # Check if correct
        if letter in self.word:
            occurrences = self.word.count(letter)
            reward = 1 * occurrences
            
            # Check if won
            if all(c in self.guessed_letters for c in self.word):
                self.won = True
                self.game_over = True
                reward += 20
        else:
            # Wrong guess
            self.wrong_guesses += 1
            reward = -5
            
            # Check if lost
            if self.wrong_guesses >= self.max_wrong_guesses:
                self.game_over = True
                reward -= 10
        
        return self.get_state(), reward, self.game_over
    
    def get_available_actions(self) -> List[str]:
        """Get unguessed letters"""
        return [c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in self.guessed_letters]


# ============================================================
# Part 3: State Encoder
# ============================================================

class StateEncoder:
    """Encode game state into feature vector"""
    
    def encode(self, state: Dict, hmm_probs: Dict[str, float]) -> np.ndarray:
        """
        Encode state into feature vector.
        
        Features:
        - Progress, lives, guessed count, word length (normalized)
        - Pattern features (vowel/consonant ratios)
        - HMM probabilities (26 dimensions)
        """
        features = []
        
        masked_word = state['masked_word']
        word_length = state['word_length']
        
        # Progress (revealed letters)
        revealed = sum(1 for c in masked_word if c != '_')
        features.append(revealed / word_length if word_length > 0 else 0)
        
        # Lives remaining
        features.append(state['lives_remaining'] / 6.0)
        
        # Guessed letters
        features.append(len(state['guessed_letters']) / 26.0)
        
        # Word length
        features.append(word_length / 20.0)
        
        # Vowel/consonant features
        vowels = sum(1 for c in masked_word if c in 'aeiou')
        consonants = revealed - vowels
        features.append(vowels / word_length if word_length > 0 else 0)
        features.append(consonants / word_length if word_length > 0 else 0)
        
        # Blank positions
        blanks = masked_word.count('_')
        features.append(blanks / word_length if word_length > 0 else 0)
        
        # HMM probabilities (26 dimensions)
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            features.append(hmm_probs.get(letter, 0.0))
        
        return np.array(features, dtype=np.float32)


# ============================================================
# Part 4: Q-Learning Agent
# ============================================================

class HangmanQLearningAgent:
    """Q-Learning agent with linear function approximation"""
    
    def __init__(self, 
                 hmm: HangmanHMM,
                 learning_rate: float = 0.01,
                 discount_factor: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995):
        
        self.hmm = hmm
        self.state_encoder = StateEncoder()
        
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Weights for Q(s, a) = w^T * phi(s, a)
        self.weights = {letter: None for letter in 'abcdefghijklmnopqrstuvwxyz'}
        
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'wins': 0,
            'losses': 0
        }
        
    def _initialize_weights(self, feature_size: int):
        """Initialize weight vectors"""
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            if self.weights[letter] is None:
                self.weights[letter] = np.random.randn(feature_size) * 0.01
    
    def get_q_value(self, state_features: np.ndarray, action: str) -> float:
        """Compute Q(s, a)"""
        if self.weights[action] is None:
            self._initialize_weights(len(state_features))
        return np.dot(self.weights[action], state_features)
    
    def choose_action(self, state: Dict, available_actions: List[str], training: bool = True) -> str:
        """Choose action using epsilon-greedy + HMM guidance"""
        hmm_probs = self.hmm.predict_letter_probs(state['masked_word'], state['guessed_letters'])
        state_features = self.state_encoder.encode(state, hmm_probs)
        
        if self.weights[available_actions[0]] is None:
            self._initialize_weights(len(state_features))
        
        # Epsilon-greedy with HMM-guided exploration
        if training and random.random() < self.epsilon:
            # Explore using HMM probabilities
            probs = [hmm_probs.get(a, 1e-6) for a in available_actions]
            total = sum(probs)
            probs = [p/total for p in probs]
            action = np.random.choice(available_actions, p=probs)
        else:
            # Exploit: Q-value + HMM boost
            action_values = {}
            for action in available_actions:
                q_val = self.get_q_value(state_features, action)
                hmm_boost = hmm_probs.get(action, 0) * 2
                action_values[action] = q_val + hmm_boost
            
            action = max(action_values.items(), key=lambda x: x[1])[0]
        
        return action
    
    def update(self, state: Dict, action: str, reward: float, next_state: Dict, done: bool):
        """Update Q-function using TD learning"""
        hmm_probs = self.hmm.predict_letter_probs(state['masked_word'], state['guessed_letters'])
        state_features = self.state_encoder.encode(state, hmm_probs)
        
        q_current = self.get_q_value(state_features, action)
        
        if done:
            q_target = reward
        else:
            next_hmm_probs = self.hmm.predict_letter_probs(
                next_state['masked_word'], 
                next_state['guessed_letters']
            )
            next_state_features = self.state_encoder.encode(next_state, next_hmm_probs)
            
            available_next = [c for c in 'abcdefghijklmnopqrstuvwxyz' 
                             if c not in next_state['guessed_letters']]
            
            if available_next:
                max_q_next = max(self.get_q_value(next_state_features, a) for a in available_next)
            else:
                max_q_next = 0
            
            q_target = reward + self.gamma * max_q_next
        
        td_error = q_target - q_current
        self.weights[action] += self.lr * td_error * state_features
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train_episode(self, word: str) -> Dict:
        """Train on single episode"""
        env = HangmanEnvironment(word)
        state = env.reset()
        
        total_reward = 0
        steps = 0
        
        while not state['game_over']:
            available_actions = env.get_available_actions()
            action = self.choose_action(state, available_actions, training=True)
            next_state, reward, done = env.step(action)
            self.update(state, action, reward, next_state, done)
            
            total_reward += reward
            steps += 1
            state = next_state
        
        self.training_stats['episodes'] += 1
        self.training_stats['total_reward'] += total_reward
        if state['won']:
            self.training_stats['wins'] += 1
        else:
            self.training_stats['losses'] += 1
        
        self.decay_epsilon()
        
        return {
            'won': state['won'],
            'total_reward': total_reward,
            'wrong_guesses': state['wrong_guesses'],
            'repeated_guesses': env.repeated_guesses,
            'steps': steps
        }
    
    def play_game(self, word: str) -> Dict:
        """Play game without training"""
        env = HangmanEnvironment(word)
        state = env.reset()
        
        total_reward = 0
        
        while not state['game_over']:
            available_actions = env.get_available_actions()
            action = self.choose_action(state, available_actions, training=False)
            next_state, reward, done = env.step(action)
            
            total_reward += reward
            state = next_state
        
        return {
            'won': state['won'],
            'total_reward': total_reward,
            'wrong_guesses': state['wrong_guesses'],
            'repeated_guesses': env.repeated_guesses,
            'word': word
        }
    
    def save(self, filename: str):
        """Save trained agent"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'training_stats': self.training_stats,
                'epsilon': self.epsilon
            }, f)
        print(f"✓ Agent saved to {filename}")
    
    def load(self, filename: str):
        """Load trained agent"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self.training_stats = data['training_stats']
            self.epsilon = data.get('epsilon', self.epsilon_end)
        print(f"✓ Agent loaded from {filename}")


# ============================================================
# Part 5: Training Pipeline
# ============================================================

def load_corpus(filename: str) -> List[str]:
    """Load word corpus from file"""
    with open(filename, 'r') as f:
        words = [line.strip().lower() for line in f if line.strip()]
    print(f"✓ Loaded {len(words)} words from {filename}")
    return words


def train_system(corpus_file: str, 
                 num_episodes: int = 1000000,
                 eval_interval: int = 1000,
                 save_prefix: str = 'hangman'):
    """Complete training pipeline"""
    
    print("=" * 60)
    print("HANGMAN AI TRAINING PIPELINE")
    print("=" * 60)
    
    # Load corpus
    print("\n1. Loading corpus...")
    words = load_corpus(corpus_file)
    
    # Train HMM
    print("\n2. Training HMM...")
    hmm = HangmanHMM(max_length=20)
    hmm.train(words)
    hmm.save(f'{save_prefix}_hmm.pkl')
    
    # Initialize agent
    print("\n3. Initializing RL agent...")
    agent = HangmanQLearningAgent(
        hmm=hmm,
        learning_rate=0.01,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    # Training loop
    print(f"\n4. Training agent ({num_episodes} episodes)...")
    print("-" * 60)
    
    for episode in range(num_episodes):
        word = random.choice(words)
        result = agent.train_episode(word)
        
        if (episode + 1) % eval_interval == 0:
            win_rate = agent.training_stats['wins'] / agent.training_stats['episodes']
            avg_reward = agent.training_stats['total_reward'] / agent.training_stats['episodes']
            
            print(f"Episode {episode + 1:5d}/{num_episodes}")
            print(f"  Win Rate:   {win_rate:.2%}")
            print(f"  Avg Reward: {avg_reward:7.2f}")
            print(f"  Epsilon:    {agent.epsilon:.4f}")
            print()
    
    # Save trained agent
    agent.save(f'{save_prefix}_agent.pkl')
    
    print("=" * 60)
    print("✓ Training complete!")
    print("=" * 60)
    
    return hmm, agent


def evaluate_system(agent: HangmanQLearningAgent, test_words: List[str]) -> Dict:
    """Evaluate agent on test set"""
    
    print("\n" + "=" * 60)
    print(f"EVALUATING ON {len(test_words)} WORDS")
    print("=" * 60)
    
    results = []
    for i, word in enumerate(test_words):
        result = agent.play_game(word)
        results.append(result)
        
        if (i + 1) % 500 == 0:
            print(f"Progress: {i+1}/{len(test_words)} games completed")
    
    # Calculate final metrics
    total_games = len(results)
    wins = sum(1 for r in results if r['won'])
    total_wrong = sum(r['wrong_guesses'] for r in results)
    total_repeated = sum(r['repeated_guesses'] for r in results)
    
    success_rate = wins / total_games
    final_score = (success_rate * total_games) - (total_wrong * 5) - (total_repeated * 2)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Games:          {total_games}")
    print(f"Wins:                 {wins} ({success_rate:.2%})")
    print(f"Losses:               {total_games - wins}")
    print(f"Total Wrong Guesses:  {total_wrong}")
    print(f"Avg Wrong/Game:       {total_wrong/total_games:.2f}")
    print(f"Total Repeated:       {total_repeated}")
    print(f"Avg Repeated/Game:    {total_repeated/total_games:.2f}")
    print("-" * 60)
    print(f"FINAL SCORE:          {final_score:.2f}")
    print("=" * 60)
    
    return {
        'total_games': total_games,
        'wins': wins,
        'success_rate': success_rate,
        'total_wrong_guesses': total_wrong,
        'total_repeated_guesses': total_repeated,
        'final_score': final_score
    }


# ============================================================
# Part 6: Main Execution
# ============================================================

if __name__ == "__main__":
    import sys
    
    # Configuration
    CORPUS_FILE = 'corpus.txt'
    TEST_FILE = 'test.txt'  # Your 2000 test words
    NUM_EPISODES = 100000
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # Training mode
        hmm, agent = train_system(
            corpus_file=CORPUS_FILE,
            num_episodes=NUM_EPISODES,
            eval_interval=1000,
            save_prefix='hangman'
        )
        
    elif len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
        # Evaluation mode
        print("Loading trained models...")
        hmm = HangmanHMM()
        hmm.load('hangman_hmm.pkl')
        
        agent = HangmanQLearningAgent(hmm)
        agent.load('hangman_agent.pkl')
        
        # Load test words
        test_words = load_corpus(TEST_FILE)
        
        # Evaluate
        results = evaluate_system(agent, test_words)
        
    else:
        print("Usage:")
        print("  python hangman_ai_full.py train      # Train the system")
        print("  python hangman_ai_full.py evaluate   # Evaluate on test set")
