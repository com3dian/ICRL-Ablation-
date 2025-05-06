from scienceworld import ScienceWorldEnv


def step(self, inputStr:str):
    observation = self.server.step(inputStr)
    raw_score = self.server.getScore()
    score = int(round(100 * raw_score))        # Convert from 0-1 to 0-100
    isCompleted = self.server.getCompleted()
    numMoves = self.get_num_moves()

    # Calculate reward
    reward = score - self.lastStepScore         # Calculate reward (delta score) for this step


    # If the number of moves exceeds the environment step limit, then set isCompleted to be true
    # if (numMoves > self.envStepLimit):
        # isCompleted = True
        # observation += "\nTask Failed. You have exceeded the maximum number of steps."

    # New: Handle this in the API rather than the agent -- if the score is less than zero, then set the isCompleted flag to true.
    if (score < 0):
        isCompleted = True
        observation += "\nTask Failed. You have done something wrong."
        reward = -100 + self.lastStepScore

    if score == 100:
        isCompleted = True
        observation += "\nTask Successfully Completed."
    
    self.lastStepScore = score                  # Store current score for reward calculation on the next step

    # Mirror of Jericho API
    infos = {
        'moves': numMoves,
        'raw_score': raw_score,
        'score': score,
        'reward': reward,
        'look': self.look(),
        'inv': self.inventory(),
        'taskDesc': self.taskdescription(),
        'valid': self.get_valid_action_object_combinations(),
        'variationIdx': self.variationIdx,
        'taskName': self.taskName,
        'simplificationStr': self.simplificationStr,
    }

    return observation, reward, isCompleted, infos


def sciworld_monkey_patch():
    ScienceWorldEnv.step = step
    print("Monkey Patched ScienceWorldEnv.step")
