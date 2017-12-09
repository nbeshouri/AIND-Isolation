"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import math
import random
from collections import defaultdict


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return -math.inf

    if game.is_winner(player):
        return math.inf

    opp_moves = game.get_legal_moves(game.get_opponent(player))
    own_moves = game.get_legal_moves(player)

    return len(own_moves) / max(len(opp_moves), 1e-6)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return -math.inf

    if game.is_winner(player):
        return math.inf

    opponent = game.get_opponent(player)

    # At the start of the game when there are lots of options,
    # use a simple heuristic.
    if len(game.get_blank_spaces()) > 25:
        opp_moves = game.get_legal_moves(opponent)
        own_moves = game.get_legal_moves(player)
        return len(own_moves) / max(len(opp_moves), 1e-6)
    # Once the board starts to fill up, use the difference between longest paths.
    else:
        return longest_path(game, player) - longest_path(game, opponent)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : isolation.Board
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return -math.inf

    if game.is_winner(player):
        return math.inf

    opponent = game.get_opponent(player)

    opp_moves = game.get_legal_moves(opponent)
    own_moves = game.get_legal_moves(player)

    # Calculate the normalized distance if both players are on the board.
    player_loc = game.get_player_location(player)
    opp_loc = game.get_player_location(opponent)
    norm_dis = 0
    if opp_loc and player_loc:
        norm_dis = distance(player_loc, opp_loc) / 8.46  # 8.46 is distance((0, 0), (6, 6))

    return len(own_moves) / max(len(opp_moves), 1e-6) - norm_dis


def try_move(game, move, player):
    """Try a move on a game with the provided player.

    This function is different from forecast_move in that it takes a player
    argument while forecast_move always moves the active player.

    Parameters
    ----------
    game : isolation.Board
        The current game.

    player : object
        The player you want to move.

    move : (int, int)
        The position to move the player.

    Returns
    -------
    isolation.Board
        A copy of the game with the player moved to the given location.
    """
    # Create a cloned game to work on.
    game_copy = game.copy()

    # Make sure that player is the current active player.
    opp = game.get_opponent(player)
    game_copy._active_player = player
    game_copy._inactive_player = opp

    game_copy.apply_move(move)
    return game_copy


def longest_path(game, player, cur_length=0, max_length=10):
    """Recursively explore the paths walkable from the player's current position
    and return the length (in moves) of the longest.

    Parameters
    ----------
    game : isolation.Board
        The game with the player on it.

    player : object
       The player to walk.

    cur_length : int (optional)
        Used to recursively calculate path length.

    max_length : int (optional)
        The largest possible path length returned. Once the this limit is
        reached, the function stops searching.

    Returns
    -------
    int
        The length of the largest path in number of moves, from 0 to `max_length`.
    """
    moves = game.get_legal_moves(player)

    # Break recursion if we reach a dead end or we're at the limit.
    if not moves or cur_length >= max_length:
        return cur_length
    # Otherwise search recursively.
    else:
        best_depth = 0
        for move in moves:
            game_copy = try_move(game, move, player)
            move_depth = longest_path(game_copy, player, cur_length + 1)
            if move_depth > best_depth:
                best_depth = move_depth
            # As soon as we're at the depth limit, there's no reason to
            # search more moves.
            if best_depth == max_length:
                return best_depth
        return best_depth


def distance(pos1, pos2):
    """Calculate the euclidean distance between two 2D points."""
    return math.sqrt((pos1[0] - pos2[0])**2. + (pos1[1] - pos2[1])**2.)


class IsolationPlayer(object):
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.minimax(game, self.search_depth)
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        moves = game.get_legal_moves(game.active_player)
        if not moves:
            return -1, -1

        scores = [(self.minimax_value(game.forecast_move(m), depth - 1), m) for m in moves]
        _, move = max(scores, key=lambda s: s[0])  # Only order on score, not move tuple.

        return move

    def minimax_value(self, game, depth):
        """Determine the value of `game` from the perspective of the player.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state.

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting.

        Returns
        -------
        score : float
            The estimated value of `game` to the player.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # If the current node is a terminal node or we've reached the depth limit,
        # return the score of the current node.
        if depth == 0 or game.is_winner(self) or game.is_loser(self):
            return self.score(game, self)

        # Otherwise return the score of the min or max child, depending on which
        # player's turn it is.
        moves = game.get_legal_moves(game.active_player)
        scores = [(self.minimax_value(game.forecast_move(m), depth - 1), m) for m in moves]
        if game.active_player == self:
            score, _ = max(scores, key=lambda s: s[0])
        else:
            score, _ = min(scores, key=lambda s: s[0])
        return score


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.

    iterative : bool (optional)
        Flag to use iterative deepening during search. Defaults to `True`.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10., iterative=True):
        super().__init__(search_depth=search_depth, score_fn=score_fn, timeout=timeout)
        self._iterative = iterative
        self._depths = []

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        if self._iterative:
            depth = 0
            try:
                while depth < len(game.get_blank_spaces()):
                    depth += 1
                    best_move = self.alphabeta(game, depth)
            except SearchTimeout:
                if depth == 1:
                    print("Warning: An iterative AlphaBetaPlayer timed out on "
                          "the first level. Match time_limit too short?")
            if depth >= 1:
                self._depths.append(depth)
        else:
            try:
                best_move = self.alphabeta(game, self.search_depth)
            except SearchTimeout:
                    print("Warning: A fixed-depth AlphaBetaPlayer timed out. "
                          "Match time_limit too short or search_depth too high.")

        return best_move

    def alphabeta(self, game, depth, alpha=-math.inf, beta=math.inf):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        moves = game.get_legal_moves(game.active_player)
        if not moves:
            return -1, -1

        scores = []
        for move in moves:
            score = self.alphabeta_value(game.forecast_move(move), depth - 1, alpha, beta)
            scores.append((score, move))
            if score > alpha:
                alpha = score

        _, best_move = max(scores, key=lambda s: s[0])
        return best_move

    def alphabeta_value(self, game, depth, alpha=-math.inf, beta=math.inf):
        """Determine the value of `game` from the perspective of the player.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state.

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting.

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        score : float
            The estimated value of `game` to the player.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # If the current node is a terminal node or we've reached the depth limit,
        # return the score of the current node.
        if depth == 0 or game.is_winner(self) or game.is_loser(self):
            return self.score(game, self)

        moves = game.get_legal_moves(game.active_player)
        scores = []

        # If this is a max node, search for the highest score.
        if game.active_player == self:

            for move in moves:
                score = self.alphabeta_value(game.forecast_move(move), depth - 1, alpha, beta)
                scores.append(score)
                # In a max node, the parent min node won't use a score
                # greater than beta, so there's no point in looking for
                # a larger one.
                if score >= beta:
                    return score
                # Update alpha so that next child I search can prune
                # itself if it knows it's going to return a lower value.
                if score > alpha:
                    alpha = score

            return max(scores)
        # If this is a min node, search for the lowest score.
        else:

            for move in moves:
                score = self.alphabeta_value(game.forecast_move(move), depth - 1, alpha, beta)
                scores.append(score)
                # In a min node, the parent max node won't use a score
                # lower than alpha, so there's no point in looking for
                # a smaller one.
                if score <= alpha:
                    return score
                # Update beta so that the next child I search can prune
                # itself if it knows it will return a higher value.
                if score < beta:
                    beta = score

            return min(scores)


class MonteCarloPlayer(IsolationPlayer):
    """
    Implement an Isolation `IsolationPlayer` that uses Monte Carlo tree search
    instead of Minimax.

    Parameters
    ----------
    score_fn : callable (optional)
        Score function used for heavy rollouts.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.

    heavy_rollouts : bool (optional)
        Whether or not to use a heuristic during rollout. If not, then moves
        are just chosen randomly until the end of the game.

    Notes
    -----
    * It wins about 30% matches against `AlphaBetaPlayer` using heavy rollouts.
    * It has not be extensively tested, so there may still be bugs hampering
    its performance.
    * It also might just be that Minimax is a better approach for this game,
    given the relatively small branching factor.
    * I don't have the `timeout` value tuned well. Using the 10 ms used in
    the other agents result in it taking too long and forfeiting. Doubling it
    mostly fixes this but probably wastes times. You can also look at inserting
    breaks in other places than at the top of the loop or profiling it to see
    why the loops take so long. Each loop should actually be pretty quick.
    """
    def __init__(self, score_fn=custom_score, timeout=20., heavy_rollouts=True):
        super().__init__(score_fn=score_fn, timeout=timeout)

        if heavy_rollouts:
            self.rollout = self.heavy_rollout
        else:
            self.rollout = self.light_rollout

        self._wins = defaultdict(lambda: 0)
        self._plays = defaultdict(lambda: 0)

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout

        best_move = (-1, -1)

        while self.time_left() > self.TIMER_THRESHOLD:
            self.update(game)

        if not self.is_terminal(game):
            moves = game.get_legal_moves()
            scores = [(self._plays[game.forecast_move(m)], m) for m in moves]
            _, best_move = max(scores, key=lambda s: s[0])

        return best_move

    def update(self, game):
        """
        Run one loop of the Monte Carlo algorithm.

        This runs a single loop of the Monte Carlo algorithm, from the current
        game state, all the way to some game ending terminal node. After each
        run, this method calls `backprop()` to update the statistics in `_wins`
        and `_plays`.

        game : isolation.Board
            The board to simulate.
        """
        cur_game = game.copy()
        visited = {cur_game}
        while not self.is_terminal(cur_game) and not self.is_unrolled_out_leaf(cur_game):
            cur_game = self.best_child(cur_game)
            visited.add(cur_game)

        # Once we get to a leaf that we need to rollout, do that. If cur_node
        # is a terminal, it will pass through rollout as is.
        terminal = self.rollout(cur_game)
        self.backprop(terminal, visited)

    def backprop(self, terminal, visited):
        """
        Use a terminal state to backpropagate the winner of a rollout.

        Parameters
        ----------
        terminal : isolation.Board
            This is the final board in the rolled out game. It must be a
            terminal node.

        visited : set
            This is the set of all "nodes" that have backpropped board to.
        """
        winner = self.get_winner(terminal)
        for game in visited:
            self._plays[game] += 1
            # Update the win count from the perspective of the inactive player
            # because that's who's making the choice to move into the this
            # state.
            if game.inactive_player == winner:
                self._wins[game] += 1

    def best_child(self, parent):
        """
        Find and return the child game with the higest UCB1

        Parameters
        ----------
        parent : isolation.Board
            The the whose children are under consideration.

        Returns
        -------
        isolation.Board
            The child with the highest UCB1 score.

        """
        child_games = [parent.forecast_move(m) for m in parent.get_legal_moves()]
        scores = [(self.ucb1(child, parent), child) for child in child_games]
        _, best = max(scores, key=lambda x: x[0])
        return best

    def light_rollout(self, game):
        """Rollout the game without using a heuristic function."""
        game = game.copy()
        while True:
            legal_player_moves = game.get_legal_moves()
            if not legal_player_moves:
                break
            game.apply_move(random.choice(legal_player_moves))
        return game

    def heavy_rollout(self, game):
        """Rollout the game with a heuristic function."""
        game = game.copy()
        while True:
            moves = game.get_legal_moves()
            if not moves:
                break
            scores = [(self.score(game.forecast_move(m), game.active_player), m) for m in moves]
            _, best = max(scores, key=lambda x: x[0])
            game.apply_move(best)
        return game

    def is_terminal(self, game):
        """Decide if a game is in a terminal (i.e. game ending) state or not."""
        return game.is_winner(game.active_player) or game.is_winner(game.inactive_player)

    def get_winner(self, game):
        """Find the winner of a completed game."""
        assert self.is_terminal(game)
        if game.is_winner(game.active_player):
            return game.active_player
        return game.inactive_player

    def is_unrolled_out_leaf(self, game):
        """Decide if a game is a leaf that needs to be rolled out."""
        return self._plays[game] == 0

    def ucb1(self, game, parent_game):
        """
        Determine a game's UCB1 score.

        Parameters
        ----------
        game : isolation.Board
            The game whose score to find.

        parent_game : isolation.Board
            The parent of the game whose score to find. This is used to find
            the total number of times the parent has been played.
        """
        assert game != parent_game
        wins = self._wins[game]
        plays = self._plays[game]
        parent_plays = self._plays[parent_game]
        assert parent_plays >= plays
        assert plays >= wins
        if not plays:
            return 1e10
        exploit = wins / plays
        explore = math.sqrt(2. * math.log(parent_plays) / plays)
        return exploit + explore
