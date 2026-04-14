 # Convert text instructions into config parameters
from nlp.interpreter import interpret_instruction 

# Train the RL agent
from rl.trainer import train_model   

 # Evaluate the trained model
from rl.evaluator import evaluate_model 

# Handle command-line arguments
import sys                                        


def main():

    # Get instruction from terminal or use default
    if len(sys.argv) > 1:
        instruction = " ".join(sys.argv[1:])
    else:
        instruction = """
        Maintain balance as long as possible,
        minimize sudden movements and prioritize stability.
        """

    print("\nINSTRUCTION:")
    print(instruction)

    # NLP → convert text into config
    config = interpret_instruction(instruction)
    print("\nGENERATED CONFIG:", config)

    # Train RL agent
    print("\nTraining agent...\n")
    model, env = train_model(config)

    # Evaluate agent
    print("\nEvaluating agent...\n")
    evaluate_model(model, env)

    print("\nProcess completed")


if __name__ == "__main__":
    main()