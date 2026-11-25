from scienceworld import ScienceWorldEnv
from py4j.protocol import Py4JJavaError
from .compat import install_scienceworld_compat


def step(self, inputStr:str):
    if not isinstance(inputStr, str):
        raise TypeError("ScienceWorld actions must be strings")
    try:
        return _step(self, inputStr)
    except Py4JJavaError as exc:
        # Py4J formats Java exceptions by calling back into the live JVM. Capture
        # that detail before the runner closes it in its cleanup block.
        raise RuntimeError(f"ScienceWorld failed executing {inputStr!r}:\n{exc}") from None


def _step(self, inputStr):
    observation = self.server.step(inputStr)
    raw_score = self.server.getScore()
    score = int(round(100 * raw_score))
    isCompleted = self.server.getCompleted()
    numMoves = self.get_num_moves()

    reward = score - self.lastStepScore 

    if (score < 0):
        isCompleted = True
        observation += "\nTask Failed. You have done something wrong."
        reward = -100 - self.lastStepScore

    if score == 100:
        isCompleted = True
        observation += "\nTask Successfully Completed."
    
    self.lastStepScore = score 

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
    install_scienceworld_compat(ScienceWorldEnv)
    ScienceWorldEnv.step = step
    print("Monkey Patched ScienceWorldEnv.step")
