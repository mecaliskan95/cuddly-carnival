import random

def guess_the_number():
    secret = random.randint(1, 10)
    print("Guess the randomly selected number between 1 and 10. You have 3 attempts.")

    for attempt in range(1, 4):
        try:
            guess = int(input(f"Attempt {attempt}: Enter your guess: "))
        except Exception as e:
            print("Error: Please enter a valid number.", e)
            continue

        if guess == secret:
            print("Congratulations! You guessed correctly!")
            return
        elif guess < secret:
            print("Try a higher number.")
        else:
            print("Try a lower number.")

    print("Sorry, you're out of attempts. The correct number was:", secret)

guess_the_number()