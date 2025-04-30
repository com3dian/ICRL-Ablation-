import re

class GSM8KTemplate:
    """
    This class helps generate prompts for grade school math word problems.
    It is inspired by methods from chain-of-thought (CoT) prompting techniques.
    The class supports:
      - Generating a prompt that includes few-shot examples.
      - Optionally enabling chain-of-thought reasoning in the prompt.
      - Extracting the final answer from a formatted example.
    """

    @staticmethod
    def generate_output(input: str, train_set: object, n_shots: int, enable_cot: bool):
        """
        Generates a prompt for a math problem.
        
        Parameters:
          input (str): The math problem to be solved.
          train_set (object): A list of training examples (dictionaries) containing 'question' and 'answer' keys.
          n_shots (int): Number of few-shot examples to include.
          enable_cot (bool): Whether to include a chain-of-thought (CoT) explanation in the prompt.
        
        Returns:
          str: A formatted prompt that includes the few-shot examples (if any) and the problem of interest.
        """
        prompt = ""
        # If few-shot prompting is used, add a header.
        if n_shots > 0:
            prompt = "The following are grade school math word problems\n"
        # Append n_shots examples from the training set.
        for i in range(n_shots):
            prompt += (
                GSM8KTemplate.format_example(train_set[i], enable_cot)
            )
        # Add the problem that we want to solve.
        # prompt += "**Problem**: " + input + "\n**Answer**: \n\n"
        prompt += "<attempt>"
        prompt += "**Problem**: " + input + "\n**Solution**: \n"
        # Depending on the flag, instruct the model to explain step-by-step or not.
        if enable_cot:
            prompt += "Let's think step-by-step."
        else:
            prompt += "No explanation needed."
        return prompt
    @staticmethod
    def generate_weak_demonstrations(input: str, weak_demo_list: list,train_set: object, n_shots: int, enable_cot: bool):
        """
        Generates a prompt for a math problem.
        
        Parameters:
          input (str): The math problem to be solved.
          train_set (object): A list of training examples (dictionaries) containing 'question' and 'answer' keys.
          n_shots (int): Number of few-shot examples to include.
          enable_cot (bool): Whether to include a chain-of-thought (CoT) explanation in the prompt.
        
        Returns:
          str: A formatted prompt that includes the few-shot examples (if any) and the problem of interest.
        """
        prompt = ""
        # If few-shot prompting is used, add a header.
        if n_shots > 0:
            prompt = "The following are grade school math word problems\n\n"
        # Append n_shots examples from the training set.
        for i in range(n_shots):
            prompt += (
                GSM8KTemplate.format_reward_example(train_set[i], enable_cot) + "\n\n"
            )
            
        # only add examples from train_set
        
        for weak_demo in weak_demo_list:
            prompt += "**Problem**: " + weak_demo['prompt'] + "\n**Reward**: "+ weak_demo['reward']+ "\n **Solution**: " + weak_demo['answer'] + "\n"
            
        
        # Add the problem that we want to solve.
        # prompt += "**Problem**: " + input + "\n**Answer**: \n\n"
        prompt +=  "**Problem**: " + input + "\n**Reward**: 10" + "\n**Solution**: \n\n"
        # Depending on the flag, instruct the model to explain step-by-step or not.
        if enable_cot:
            prompt += "Let's think step-by-step."
        else:
            prompt += "No explanation needed."
        return prompt
    @staticmethod
    def format_reward_example(data: dict, enable_cot: bool):
        """
        Formats a training example.
        
        The example dictionary is expected to have:
          - 'question': The problem statement.
          - 'answer': A string containing both the solution explanation and the final answer,
                      separated by "\n#### " (i.e., a newline, four hash symbols, and a space).
        
        Parameters:
          data (dict): A dictionary containing a training example.
          enable_cot (bool): Whether to include the detailed solution explanation.
        
        Returns:
          str: A formatted string showing the problem, solution (if enabled), and answer.
        """
        formatted_problem = ""
        question = data["question"]
        formatted_problem += "**Problem**: " + question + "\n" + "**Reward**: 10" + "\n"
        raw_answer = data["answer"]
        # Split the answer into the solution explanation and the final answer.
        solution, answer = raw_answer.strip().split("\n#### ")
        if enable_cot:
            formatted_problem += "**Solution**: " + solution + "\n"
        formatted_problem += "**Answer**: " + answer
        return formatted_problem

    @staticmethod
    def format_example(data: dict, enable_cot: bool):
        """
        Formats a training example.
        
        The example dictionary is expected to have:
          - 'question': The problem statement.
          - 'answer': A string containing both the solution explanation and the final answer,
                      separated by "\n#### " (i.e., a newline, four hash symbols, and a space).
        
        Parameters:
          data (dict): A dictionary containing a training example.
          enable_cot (bool): Whether to include the detailed solution explanation.
        
        Returns:
          str: A formatted string showing the problem, solution (if enabled), and answer.
        """
        formatted_problem = "<attempt>\n"
        question = data["question"]
        formatted_problem += "**Problem**: " + question + "\n"
        raw_answer = data["answer"]
        # Split the answer into the solution explanation and the final answer.
        solution, answer = raw_answer.strip().split("\n#### ")
        if enable_cot:
            formatted_problem += "**Solution**: " + solution + "\n"
        formatted_problem += "**Answer**: " + answer
        formatted_problem += "\n</attempt>\n"
        return formatted_problem

    @staticmethod
    def format_answer(data: dict):
        """
        Extracts and returns the final answer from a training example.
        
        It uses a regular expression to find the text after "#### " in the answer string.
        
        Parameters:
          data (dict): A dictionary containing a training example.
        
        Returns:
          str: The extracted final answer.
        """
        raw_answer = data["answer"]
        answer = re.findall(r"#### (.*)", raw_answer)[0]
        return answer

    @staticmethod
    def format_subject(subject: str):
        """
        Placeholder method for formatting a subject.
        Currently not implemented.
        
        Parameters:
          subject (str): A subject string.
        
        Returns:
          str: The subject unchanged.
        """
        return subject  # This method can be extended as needed.