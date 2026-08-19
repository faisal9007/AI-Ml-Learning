age = int(input("Enter your age: "))

if age >= 18:
    print("You are an adult.")
else:
    print("You are a minor.")

secret = 7

guess = int(input("Guess a number between 1 and 10: "))

if guess == secret:
    print("🎉 You Win!")
else:
    print("❌ Try Again!")