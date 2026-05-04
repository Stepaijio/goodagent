# Main entry point for the project
from src.physics.visualizer import animate_film
from src.data_gen.generator import generate_dataset

if __name__ == "__main__":
    print("Select mode:")
    print("1 - Generate Dataset")
    print("2 - Visualize Film")
    
    choice = input("Enter choice (1 or 2): ")

    if choice == "1":
        print("Starting Dataset Generation...")
        generate_dataset()
    elif choice == "2":
        print("Starting Kapitza Waves Visualization...")
        animate_film()
    else:
        print("Invalid choice. Please run the program again and select 1 or 2.")
