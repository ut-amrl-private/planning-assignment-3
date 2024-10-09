# **Assignment 3: Adversarial Search**

In this assignment, you will implement adversarial search methods to solve a two-player zero-sum board game called [End of the Track](https://www.gaya-game.com/products/the-end-of-the-track) using Python. Please refer to Assignment 2 for the rules of the game. **You will extend your work from Assignment 2**.

This is a rather open-ended assignment:

Your goal for this assignment is to implement one adversarial search method (of your choice). Your algorithm will need to employ a heuristic value function (heuristic to evaluate the value of a state or state-action).

You may wish to consider the following (though some may not be able to solve the game): minimax, alpha-beta pruning, cutoffs, iterative deepening, pruning, caching, cycle detection, use of heuristics or evaluation functions, Monte-Carlo Tree Search, using rollouts, ordering of moves, etc; or interesting combinations of the above.

**Evaluation:**

The methods you implement should be able to solve games against:

- A random player (a player that takes random actions)
- A passive player (one that simply moves their ball around without moving any of their block pieces)
- An optimal player

We will test your submission on test cases with guaranteed outcomes under optimal play; your algorithm should be capable of playing for either white or black, starting from an arbitrary board configuration. We will grade outcomes based on at most 4-steps from termination.

**Extra Credit:**

The assignment is worth 195 points in total. There an an additional 3 extra credit points. We test up to a horizon of 3 total moves to a win. One of the tests is worth 3 points extra credit.

**Implementation Guidelines:**

Please follow the instructions on the next page with regards to your implementation and submission. In particular, we will only grade the player class called AdversarialSearchPlayer located in the game.py file. Your adversarial algorithms should be added to search.py under GameStateProblem; your players will then make a call to the appropriate algorithm.

**Code Handout and Reference:**

We recommend starting from a copy of your work from Assignment 2.

**Please add the following Player class to your game.py**: This will serve as the base class from which you will subclass to create different types of players that play the game. Here, policy_fnc corresponds to the adversarial search algorithm the player will use to play the game.
```python
class Player:
    def __init__(self, policy_fnc):
        self.policy_fnc = policy_fnc

    def policy(self, decode_state):
        pass
```
You can create a specific type of Player in `game.py` by subclassing from Player. You are welcome to create multiple types of Players, **but the one we grade will be called AdversarialSearchPlayer, and it has to be in game.py**
```python
class AdversarialSearchPlayer(Player):
    def __init__(self, gsp, player_idx):
        """
        You can customize the signature of the constructor above to suit your needs.
        In this example, in the above parameters, gsp is a GameStateProblem, and
        gsp.adversarial_search_method is a method of that class.  
        """
        super().__init__(gsp.adversarial_search_method)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx

    def policy(self, decode_state):
        """
        Here, the policy of the player is to consider the current decoded game state
        and then correctly encode it and provide any additional required parameters to the
        assigned policy_fnc (which in this case is gsp.adversarial_search_method), and then
        return the result of self.policy_fnc
        Inputs:
          - decoded_state is a 12-tuple of ordered pairs. For example:
          (
            (c1,r1),(c2,r2),(c3,r3),(c4,r4),(c5, r5),(c6,r6),
            (c7,r7),(c8,r8),(c9,r9),(c10,r10),(c12,r12),(c12,r12),
          )
        Outputs:
          - policy returns a tuple (action, value), where action is an action tuple
          of the form (relative_idx, encoded_position), and value is a value.
        NOTE: While value is not used by the game simulator, you may wish to use this value
          when implementing your policy_fnc. The game simulator and the tests only call
          policy (which wraps your policy_fnc), so you are free to define the inputs for policy_fnc.
        """
        encoded_state_tup = tuple( self.b.encode_single_pos(s) for s in decode_state )
        state_tup = tuple((encoded_state_tup, self.player_idx))
        val_a, val_b, val_c = (1, 2, 3)
        return self.policy_fnc(state_tup, val_a, val_b, val_c)
```

**You can add multiple adversarial search algorithms to the GameStateProblem class, and then create various Player classes which use those specific algorithms**. Make sure that when you call `self.policy_fnc`, the function signature matches the signature of the function that is the entry to the algorithm that the player will be using.

**Note** that in the `GameStateProblem `class, there is a distinction between checking if a state is a goal state (Problem.is_goal) and checking if a state is a terminal state: `GameStateProblem.is_termination_state` uses `BoardState.is_termination_state`, which you implemented in Assignment 2. The former is used to check if an arbitrary state has been reached, whereas the latter checks if a state corresponds to a player having won the game.

**Tabulating Experiment Parameters in Pytest:**

To tabulate your experiments, you can create new file **test_game.py** and add the following parameterized test to the **test_game.py** under the  `TestGame class`, using **@pytest.mark.parametrize**. See code block below.
```python
@pytest.mark.parametrize("p1_class, p2_class, encoded_state_tuple,exp_winner,exp_stat",
    [(PlayerAlgorithmA, PlayerAlgorithmB, (49, 37, 46, 41, 55, 41, 50, 51, 52, 53, 54, 52), "WHITE", "No issues")])
def test_adversarial_search(self, p1_class, p2_class, encoded_state_tuple, exp_winner, exp_stat):
    b1 = BoardState()
    b1.state = np.array(encoded_state_tuple)
    b1.decode_state = b1.make_state()
    players = [p1_class(GameStateProblem(b1, b1, 0), 0), p2_class(GameStateProblem(b1, b1, 0), 1)]
    sim = GameSimulator(players)
    sim.game_state = b1
    rounds, winner, status = sim.run()
    assert winner == exp_winner and status == exp_stat
```
The above example assumes that two player classes, PlayerAlgorithmA and PlayerAlgorithmB have been created. An encoded game state has been provided where the first player (player_idx = 0) is WHITE, and is 1 move away from winning, if the player plays optimally. Therefore in the assert statement, we expect WHITE to win, and no errors to occur.

You can tabulate your experiments by adding more tuples (also feel free to add in additional parameters that you might need for your algorithms, or separate test functions if you feel those would better capture your experiment settings). Here are small subset of tuples which Autograde will check. There is no need to modify `GameSimulator.run`, if you modified for your local test or debug, please revert it back before you submit your solution
```python
(Your AdversarialSearchPlayer, PlayerWithAlgorithmB, (49, 37, 46,  7, 55,  7, 50, 51, 52, 53, 54, 52), "WHITE", "No issues") 
(Your AdversarialSearchPlayer, PlayerWithAlgorithmB, (49, 37, 46,  0, 55,  0, 50, 51, 52, 53, 54, 52), "WHITE", "No issues") 
(PlayerWithAlgorithmB, Your AdversarialSearchPlayer, (14, 21, 22, 28, 29, 22,  9, 20, 34, 39, 55, 55), "BLACK", "No issues") 
(PlayerWithAlgorithmB, Your AdversarialSearchPlayer, (14, 21, 22, 28, 29, 22, 11, 20, 34, 39, 55, 55), "BLACK", "No issues") 
(Your AdversarialSearchPlayer, PlayerWithAlgorithmB, (44, 37, 46, 34, 40, 34,  1,  2, 52,  4,  5, 52), "WHITE", "No issues") 
(Your AdversarialSearchPlayer, PlayerWithAlgorithmB, (44, 37, 46, 28, 40, 28,  1,  2, 52,  4,  5, 52), "WHITE", "No issues") 
```

In order to run only this specific test using Pytest, you can run pytest -k test_adversarial_search

From the Pytest manual (pytest --help), you can use -k to filter running specific tests, please refer https://docs.pytest.org/en/stable/how-to/usage.html#specifying-which-tests-to-run for details.

**Tips for Debugging:**

- If you are debugging and wish to see printed outputs, refer to the Pytest manual. This post may also be helpful:  
    <https://stackoverflow.com/questions/14405063/how-can-i-see-normal-print-output-created-during-pytest-run>
- You can also take the body of test_adversarial_search and execute it separately.
- It may be helpful to print out / record values of states, actions taken, paths explored, etc along with which player is performing the search.

# Submission Instructions
Please submit a zip file in Gradescope under AI388U-assignment3. Your zip file should include the following files only (no nested folders, no other hidden files):
- game.py
- search.py
- helper files if you have any (no test files should be included) 
